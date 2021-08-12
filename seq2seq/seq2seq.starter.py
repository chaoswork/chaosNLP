#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import re
import time
import json
import argparse
import unicodedata

import jieba
import numpy as np
import tensorflow as tf

from opencc import OpenCC
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu


os.environ["CUDA_VISIBLE_DEVICES"] = '3'



table = {ord(f):ord(t) for f,t in zip(
     u'，。！？【】（）％＃＠＆１２３４５６７８９０',
     u',.!?[]()%#@&1234567890')}

class NMTDataset(object):
    def __init__(self, file_path):
        # self.problem_type = '-spa'
        self.file_path = file_path
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None

    @staticmethod
    def preprocess_sentence(w, sent_type):
        if sent_type == 'chn':
            w = OpenCC('t2s').convert(w)
            w = ' '.join(list(jieba.cut(w)))
        w = w.lower().translate(table)
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = '<start> ' + w.strip() + ' <end>'
        return w

    @staticmethod
    def create_dataset(path, num_examples=None):
        # path : path to chn-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[NMTDataset.preprocess_sentence(w, s_type) for w, s_type in zip(l.split('\t')[1::2], ['chn', 'eng'])]  for l in lines[:num_examples]]

        return zip(*word_pairs)

    def tokenize(self, lang):
        # lang = list of sentences in a language
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(lang)

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn)
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(lang)

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences
        ## and pads the sequences to match the longest sequences in the given input
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, path, num_examples=None):
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset(path, num_examples)
        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def call(self, BUFFER_SIZE, BATCH_SIZE, num_examples=None):
        file_path = self.file_path
        input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(file_path, num_examples)


        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


        return train_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer


#####

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        ##________ LSTM layer in Encoder ------- ##
        self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')



    def call(self, x, hidden):
        # x: (batch_size, time)
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state = hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]




class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units,
                 batch_sz, attention=None):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.lstm_cell = tf.keras.layers.LSTMCell(self.dec_units)

        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = None
        if attention == 'Bahdanau':
            self.attention = BahdanauAttention(self.dec_units)

    def one_step(self, inputs, state, enc_outputs):
        # inputs.shape = (batch_sz, 1)
        # x.shape = (batch_sz, emb_dim)
        x = tf.squeeze(self.embedding(inputs), axis=1)
        # print(x.shape, state[0].shape, state[1].shape)

        if self.attention:
            context_vector, attention_weights = self.attention(state[0], enc_outputs)
            x = tf.concat([x, context_vector], axis=1)

        output, states = self.lstm_cell(x, state)
        output = tf.reshape(output, [-1, self.dec_units])
        output = self.fc(output)
        return output, states

    def call(self, inputs, initial_state, enc_outputs):
        # inputs: (batch_size, feature_size)
        # initial_state: list of (batch_size, units)
        # enc
        outputs = []
        total_steps = inputs.shape[1]
        states = initial_state
        for i in range(total_steps):
            input_ti = inputs[:, i]
            # print(input_ti.shape)
            output, states = self.one_step(input_ti, states, enc_outputs)
            outputs.append(output)
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, total_steps, self.vocab_size])
        return outputs


# Attention Mechanism
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size) s_i in paper
        # values shape == (batch_size, max_len, hidden size) h in paper, h_j is value[:,j,:]

        # we are doing this to broadcast addition along the time axis to calculate the score
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        # score[:,j,:] is e_ij = a(s_{i-1}, h_j)
        # a = tanh(s_{i-1} x W1 + h_j x W2 ) x W_{units, 1}

        # attention_weights shape == (batch_size, max_length, 1)
        # \alpha_{ij} in paper
        # attention_weights[:,j,:] = \alpha_{ij}
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class NMT(object):
    def __init__(self, checkpoint_dir, word_embedding_dim=256, units=1024,
                 buffer_size=32000, batch_size=64, epochs=5,
                 attention=None, num_examples=None, config_file='nmt.config',
                 is_train=False, train_file=None, is_eval=False, test_file=None,
                ):

        self.word_embedding_dim = word_embedding_dim
        self.units = units
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.beam_size = 3
        self.epochs = epochs
        self.num_examples = num_examples
        self.checkpoint_dir = checkpoint_dir
        self.attention = attention
        self.config_file = config_file
        self.is_train = is_train
        self.train_file = train_file
        if self.is_train:
            self.build_dataset(train_file)
            self.build_model()
            self.train()
        else:
            self.restore_config()
            self.build_model()
            self.restore_weights()

        if is_eval:
            self.evaluate_data(test_file)



    def restore_config(self, config_file=None):
        if config_file is None:
            config_file = self.config_file

        lines = io.open(config_file).read().strip()
        config = json.loads(lines)
        self.inp_lang = tf.keras.preprocessing.text.tokenizer_from_json(
            config['inp_lang_tokenizer'])
        self.targ_lang = tf.keras.preprocessing.text.tokenizer_from_json(
            config['targ_lang_tokenizer'])
        self.vocab_inp_size = len(self.inp_lang.word_index)+1
        self.vocab_tar_size = len(self.targ_lang.word_index)+1
        self.max_length_input = config['max_length_input']
        self.max_length_output = config['max_length_output']

    def save_config(self, config_file):
        config = {}
        config['inp_lang_tokenizer'] = self.inp_lang.to_json()
        config['targ_lang_tokenizer'] = self.targ_lang.to_json()
        config['max_length_input'] = self.max_length_input
        config['max_length_output'] = self.max_length_output

        with open(config_file, 'w') as f:
            f.write(json.dumps(config, ensure_ascii=False))


    def build_dataset(self, file_path):

        self.dataset_creator = NMTDataset(file_path)
        self.train_dataset, self.inp_lang, self.targ_lang = \
            self.dataset_creator.call(self.buffer_size, self.batch_size, self.num_examples)

        self.vocab_inp_size=len(self.inp_lang.word_index)+1
        self.vocab_tar_size=len(self.targ_lang.word_index)+1
        example_input_batch, example_target_batch = next(iter(self.train_dataset))
        self.max_length_input=example_input_batch.shape[1]
        self.max_length_output=example_target_batch.shape[1]

        self.save_config(self.config_file)

    def build_model(self):
        self.encoder = Encoder(self.vocab_inp_size, self.word_embedding_dim,
                               self.units, self.batch_size)
        self.decoder = Decoder(self.vocab_tar_size, self.word_embedding_dim,
                               self.units, self.batch_size, attention=self.attention)

        self.optimizer = tf.keras.optimizers.Adam()

        # checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                 encoder=self.encoder,
                                 decoder=self.decoder)

    def loss_function(self, real, pred):
        # real shape = (BATCH_SIZE, max_length_output)
        # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)
        mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask* loss
        loss = tf.reduce_mean(loss)
        return loss

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)


            dec_input = targ[ : , :-1 ] # Ignore <end> token
            real = targ[ : , 1: ]         # ignore <start> token

            # Set the AttentionMechanism object with encoder_outputs
            # decoder.attention_mechanism.setup_memory(enc_output)

            # Create AttentionWrapperState as initial_state for decoder
            # decoder_initial_state = decoder.build_initial_state(BATCH_SIZE, [enc_h, enc_c], tf.float32)
            # pred = decoder(dec_input, decoder_initial_state)
            decoder_initial_state = [enc_h, enc_c]
            pred = self.decoder(dec_input, decoder_initial_state, enc_output)
            logits = pred
            loss = self.loss_function(real, logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def train(self, train_dataset=None):
        if not train_dataset:
            train_dataset = self.train_dataset
        # EPOCHS = 10

        for epoch in range(self.epochs):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            # print(enc_hidden[0].shape, enc_hidden[1].shape)

            for (batch, (inp, targ)) in enumerate(train_dataset):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                           batch,
                                                           batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            steps_per_epoch = batch + 1

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                              total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            self.translate("How old are you ?")


    def restore_weights(self, checkpoint_dir=None):
        if not checkpoint_dir:
            checkpoint_dir = self.checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def decode_beam_search(self, sentences):

        sentences = [NMTDataset.preprocess_sentence(line, 'eng') for line in sentences]

        inputs = self.inp_lang.texts_to_sequences(sentences)
        #print(inputs)
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                          maxlen=self.max_length_input,
                                                          padding='post')
        #print(inputs)

        # Convert the inputs to tensors
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]

        result = '<start> '
        enc_start_state = [tf.zeros((inference_batch_size, self.units)),
                           tf.zeros((inference_batch_size, self.units))]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)
        # print(enc_h.shape, enc_c.shape)
        decoder_initial_state = [enc_h, enc_c]
        # dec_input: (batch_size, 1)
        dec_input = tf.reshape([self.targ_lang.word_index['<start>']] * inference_batch_size, [-1,1])

        # dec_input: (batch_size, beam_size, 1)
        dec_input = tf.tile(tf.expand_dims(dec_input, axis=1), [1, self.beam_size, 1])
        # TODO: debug
#         dec_input = tf.reshape(list(range(1, 1+ self.beam_size * inference_batch_size)),
#                                [inference_batch_size, self.beam_size, 1]
#                               )
        print('dec_input.shape:', dec_input.shape)

        beam_state_list = [decoder_initial_state] * self.beam_size
        beam_log_probs = tf.zeros((inference_batch_size, self.beam_size, 1))
        # (batch_size, beam_size, 1)
        beam_path_list = tf.tile(
            tf.reshape([self.targ_lang.word_index['<start>']], [1,1,1]),
            [inference_batch_size, self.beam_size, 1])
        # TODO: debug
        # beam_path_list = dec_input
        print(dec_input.shape, decoder_initial_state[0].shape, decoder_initial_state[1].shape)
        finished_buffer_size = self.beam_size * self.beam_size
        finished_path = np.zeros((inference_batch_size, finished_buffer_size, self.max_length_output), dtype=int)
        finished_probs = np.zeros((inference_batch_size, finished_buffer_size))
        finished_counts = [0] * inference_batch_size
        for t in range(self.max_length_output):
            # print(dec_input.shape)
            # TODO: 当前还是取下标的方式，后续改成batch_size * beam_size的方式
            bfs_result = []
            for bid in range(self.beam_size):
                # dec_input: (batch_size, 1)
                inputs = dec_input[:,bid,:]
                #print(t, bid, inputs)
                logits, state = self.decoder.one_step(inputs, beam_state_list[bid], enc_out)
                #print(logits.shape)
                log_probs = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)

                log_probs += beam_log_probs[:,bid]
                # topk_indexs = np.argpartition(log_probs, -beam_size)[:,-beam_size:]
                topk_values, topk_indices = tf.nn.top_k(log_probs, self.beam_size)
                print("Time {}, iter beam {}".format(t, bid))
                print('  topk_values', topk_values)
                print('  topk_indices', topk_indices)
                bfs_result.append((state, topk_values, topk_indices))




                for batch_id in range(topk_indices.shape[0]):
                    print(' input word', self.targ_lang.index_word[inputs[batch_id, 0].numpy()])
                    for predicted_id in topk_indices[batch_id]:
        #                 if predicted_id not in self.targ_lang.index_word:
        #                     continue
                        print('\t', predicted_id, self.targ_lang.index_word[predicted_id.numpy()],
                              np.exp(log_probs[batch_id][predicted_id]))
                    print('-' * 10)
                # 对于start只需要search一次。
                if t == 0:
                    break
            # END OF beam loop
            # 构造新的输入
            beam_log_probs = tf.concat([x[1] for x in bfs_result], axis=1)
            beam_pred_ids = tf.concat([x[2] for x in bfs_result], axis=1)


            # handle path
            if t > 0:
                beam_path_list_expand = tf.reshape(
                    tf.tile(beam_path_list, [1, 1, self.beam_size]),
                    [inference_batch_size, self.beam_size * self.beam_size, -1])
                beam_state_list = [x[0] for x in bfs_result]
            else:
                beam_path_list_expand = beam_path_list
                beam_state_list = [bfs_result[0][0]] * self.beam_size
                # [inference_batch_size, self.beam_size * self.beam_size, -1])
            new_path = tf.concat([beam_path_list_expand, tf.expand_dims(beam_pred_ids, -1)], axis=-1)

           # find finished result
            finished_indices = tf.where(beam_pred_ids == nmt.targ_lang.word_index['<end>'])

            if finished_indices.shape[0] > 0:
                print('handle finised, finished_indices', finished_indices)
                for i in range(finished_indices.shape[0]):
                    batch_id, end_id = finished_indices[i].numpy()
                    if finished_counts[batch_id] >= finished_buffer_size:
                        continue
                    # print('debug', batch_id, end_id, finished_counts[batch_id] )
                    finished_probs[batch_id][finished_counts[batch_id]] = \
                        np.exp(beam_log_probs[batch_id][end_id].numpy())
                    cur_finished_path = new_path[batch_id][end_id].numpy()
                    for j in range(len(cur_finished_path)):
                        finished_path[batch_id][finished_counts[batch_id]][j] = cur_finished_path[j]
                    finished_counts[batch_id] += 1
                # 将已完成的概率降权，使topk无法取到。
                print('debug', beam_log_probs)
                beam_log_probs = beam_log_probs + \
                    tf.cast(beam_pred_ids == nmt.targ_lang.word_index['<end>'], tf.float32)\
                        * tf.reduce_min(beam_log_probs)

                print('finished_probs:', finished_probs)
                print('finished_path:', finished_path)


            # get top k gather index
            values, indices = tf.nn.top_k(beam_log_probs, self.beam_size)
            beam_log_probs = tf.expand_dims(values, -1)


            rows = tf.tile(
                tf.reshape(tf.range(inference_batch_size), [inference_batch_size, 1, 1]),
                [1, self.beam_size, 1])
            gather_index = tf.concat([rows, tf.expand_dims(indices, -1)], axis=-1)


            dec_input = tf.expand_dims(tf.gather_nd(beam_pred_ids, gather_index), -1)
            beam_path_list = tf.gather_nd(new_path, gather_index)
            # print('beam_path_list_expand:', beam_path_list_expand)
            print('new_path:', beam_path_list)
            print('beam best translate:')
            for batch_id in range(beam_path_list.shape[0]):
                for beam_id in range(beam_path_list.shape[1]):
                    line = ' '.join([self.targ_lang.index_word[predict_id.numpy()] \
                     for predict_id in beam_path_list[batch_id][beam_id]])
                    print('batch_id {}, beam_id {}, prob {}, {}'.format(
                        batch_id, beam_id,
                        np.exp(beam_log_probs[batch_id][beam_id][0].numpy()),
                        line))
            if sum(finished_counts) == len(finished_counts) * finished_buffer_size:
                print('buffer full filled')
                break
        final_probs = []
        final_result = []
        for batch_id, max_index in enumerate(np.argmax(finished_probs, axis=1)):
            final_probs.append(finished_probs[batch_id][max_index])
            final_result.append(' '.join(
                [self.targ_lang.index_word[wid]\
                    for wid in filter(
                        lambda x: x > 0, finished_path[batch_id][max_index])]))

        return final_result, final_probs, finished_probs, finished_path


            # return bfs_result
            # beam_path_list = [x[0] for x in bfs_result]
            # break
            #dec_input = tf.reshape([predicted_id], [-1,])

#         # 初始概率为0
#         beam_size = 3
#         initial_log_probs = tf.constant([[0.] + [-INF] * (beam_size - 1)])
#         alive_log_probs = tf.tile(initial_log_probs, [inference_batch_size, 1])

#         def _expand_to_beam_size(tensor, beam_size):
#             """Tiles a given tensor by beam_size.
#             Args:
#             tensor: tensor to tile [batch_size, ...]
#             beam_size: How much to tile the tensor by.
#             Returns:
#             Tiled tensor [batch_size, beam_size, ...]
#             """
#             tensor = tf.expand_dims(tensor, axis=1)
#             tile_dims = [1] * tensor.shape.ndims
#             tile_dims[1] = beam_size

#             return tf.tile(tensor, tile_dims)

#         # Expand each batch and state to beam_size
#         initial_ids = dec_input
#         # (inference_batch_size, beam_size)
#         alive_seq = _expand_to_beam_size(initial_ids, beam_size)
#         alive_seq = tf.expand_dims(alive_seq, axis=2)  # (batch_size, beam_size, 1)



    def decode_greedy(self, sentence):

        # Attention plot (to be plotted later on) -- initialized with max_lengths of both target and input
        # attention_plot = np.zeros((max_length_output, max_length_input))

        # Preprocess the sentence given
        # sentence = preprocess_sentence(sentence)
        sentence = NMTDataset.preprocess_sentence(sentence, 'eng')


        # Fetch the indices concerning the words in the sentence and pad the sequence
        # inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = self.inp_lang.texts_to_sequences([sentence])[0]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                          maxlen=self.max_length_input,
                                                          padding='post')

        # Convert the inputs to tensors
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]

        result = '<start> '
        enc_start_state = [tf.zeros((inference_batch_size, self.units)),
                           tf.zeros((inference_batch_size, self.units))]
        print('debug', inputs.shape)
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        decoder_initial_state = [enc_h, enc_c]
        # dec_input: ()
        dec_input = tf.reshape([self.targ_lang.word_index['<start>']], [-1,1])
        print(dec_input.shape)

        # Loop until the max_length is reached for the target lang (ENGLISH)
        state = decoder_initial_state
        for t in range(self.max_length_output):
            # print(dec_input.shape)
            predictions, state = self.decoder.one_step(dec_input, state, enc_out)
    #         predictions, dec_hidden, attention_weights = decoder(dec_input,
    #                                                              dec_hidden,
    #                                                              enc_out)

            # Store the attention weights to plot later on
    #         attention_weights = tf.reshape(attention_weights, (-1, ))
    #         attention_plot[t] = attention_weights.numpy()

            # Get the prediction with the maximum attention
            predicted_id = tf.argmax(predictions[0]).numpy()
            # print (predicted_id)

            # Append the token to the result
            result += self.targ_lang.index_word[predicted_id] + ' '

            # If <end> token is reached, return the result, input, and attention plot
            if self.targ_lang.index_word[predicted_id] == '<end>':
                return result

            # The predicted ID is fed back into the model
            dec_input = tf.reshape([predicted_id], [-1,1])

        return result# , sentence# , attention_plot

    def translate(self, sentence):
        result = self.decode_greedy(sentence)
        # print(result)
        # result = self.targ_lang.sequences_to_texts(result)
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))
        return result

    def evaluate_data(self, test_file):
        targets, inputs = NMTDataset.create_dataset(test_file)
        bleu_score = 0
        targets_len = len(targets)
        for i in range(targets_len):
            hypothesis = self.translate(inputs[i]).strip().split()[1:-1]
            reference = [targets[i].strip().split()[1:-1]]
            print(hypothesis, reference)
            print(targets[i], inputs[i], self.translate(inputs[i]))
            bleu_score += sentence_bleu(reference, hypothesis)
        print("BLEU Score: {}".format(bleu_score / targets_len))




parser = argparse.ArgumentParser(description='Process NMT parameters.')
parser.add_argument('--is_train', dest='is_train', action='store_true')
parser.add_argument('--train_file', dest='train_file', default=None)
parser.set_defaults(is_train=False)

parser.add_argument('--is_eval', dest='is_eval', action='store_true')
parser.add_argument('--test_file', dest='test_file', default=None)
parser.set_defaults(is_eval=False)

parser.print_help()
args = parser.parse_args()


# nmt = NMT(checkpoint_dir='./training_attention_checkpoints',
#           attention='Bahdanau')
# nmt.translate("How old are you ?")

nmt = NMT(checkpoint_dir='./training_attention_checkpoints',
          num_examples=1000,
          epochs=2,
          attention='Bahdanau',
          is_train=args.is_train,
          train_file=args.train_file,
          is_eval=args.is_eval,
          test_file=args.test_file
         )

result = nmt.decode_beam_search(["How old are you ?",
                                 "What is your name ?",
                                 "She's a nurse",
                                 "I received your letter.",
                                ])
print(result)

nmt.translate("How old are you ?")

