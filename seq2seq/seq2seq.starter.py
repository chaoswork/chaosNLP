#-*- coding:utf8 -*-

import io
import os
import re
import time
import unicodedata

import jieba
import tensorflow as tf
from opencc import OpenCC
from sklearn.model_selection import train_test_split

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

    def preprocess_sentence(self, w, sent_type):
        if sent_type == 'chn':
            w = OpenCC('t2s').convert(w)
            # w = ' '.join(w) # TODO：可以考虑切词
            w = ' '.join(list(jieba.cut(w)))
        w = w.lower().translate(table)
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = '<start> ' + w.strip() + ' <end>'
        return w

    def create_dataset(self, path, num_examples=None):
        # path : path to chn-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[self.preprocess_sentence(w, s_type) for w, s_type in zip(l.split('\t')[1::2], ['chn', 'eng'])]  for l in lines[:num_examples]]

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

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer


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
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.lstm_cell = tf.keras.layers.LSTMCell(self.dec_units)

        self.fc = tf.keras.layers.Dense(vocab_size)

    def one_step(self, inputs, state):
        # inputs.shape = (batch_sz, )
        # x.shape = (batch_sz, emb_dim)
        x = self.embedding(inputs)
        # output.shape = (batch_sz, 1, dec_units)
        # print(x.shape, state[0].shape, state[1].shape)
        output, states = self.lstm_cell(x, state)
        output = tf.reshape(output, [-1, self.dec_units])
        output = self.fc(output)
        return output, states

    def call(self, inputs, initial_state):
        outputs = []
        total_steps = inputs.shape[1]
        states = initial_state
        for i in range(total_steps):
            input_ti = inputs[:, i]
            # print(input_ti.shape)
            output, states = self.one_step(input_ti, states)
            outputs.append(output)
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, total_steps, self.vocab_size])
        return outputs





class NMT(object):
    def __init__(self,
                 checkpoint_dir,
                 word_embedding_dim=256,
                 units=1024,
                 buffer_size=32000,
                 batch_size=64,
                 epochs=5,
                 num_examples=None
                ):

        self.word_embedding_dim = word_embedding_dim
        self.units = units
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_examples = num_examples
        self.checkpoint_dir = checkpoint_dir




    def build_dataset(self, file_path):

        self.dataset_creator = NMTDataset(file_path)
        self.train_dataset, self.val_dataset, self.inp_lang, self.targ_lang = \
            self.dataset_creator.call(self.buffer_size, self.batch_size, self.num_examples)

        self.vocab_inp_size=len(self.inp_lang.word_index)+1
        self.vocab_tar_size=len(self.targ_lang.word_index)+1
        example_input_batch, example_target_batch = next(iter(self.train_dataset))
        self.max_length_input=example_input_batch.shape[1]
        self.max_length_output=example_target_batch.shape[1]

    def build_model(self):
        self.encoder = Encoder(self.vocab_inp_size, self.word_embedding_dim,
                               self.units, self.batch_size)
        self.decoder = Decoder(self.vocab_tar_size, self.word_embedding_dim,
                               self.units, self.batch_size)

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
            pred = self.decoder(dec_input, decoder_initial_state)
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
        steps_per_epoch = self.num_examples//self.batch_size

        for epoch in range(self.epochs):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            # print(enc_hidden[0].shape, enc_hidden[1].shape)

            for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                           batch,
                                                           batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                              total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            self.translate("How old are you ?")


    def restore_weights(self, checkpoint_dir=None):
        if not checkpoint_dir:
            checkpoint_dir = self.checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def evaluate_greedy(self, sentence):

        # Attention plot (to be plotted later on) -- initialized with max_lengths of both target and input
        # attention_plot = np.zeros((max_length_output, max_length_input))

        # Preprocess the sentence given
        # sentence = preprocess_sentence(sentence)
        sentence = self.dataset_creator.preprocess_sentence(sentence, 'eng')


        # Fetch the indices concerning the words in the sentence and pad the sequence
        inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                          maxlen=self.max_length_input,
                                                          padding='post')

        # Convert the inputs to tensors
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]

        result = ''
        enc_start_state = [tf.zeros((inference_batch_size, self.units)),
                           tf.zeros((inference_batch_size, self.units))]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        decoder_initial_state = [enc_h, enc_c]

        dec_input = tf.reshape([self.targ_lang.word_index['<start>']], [-1,])
        # print(dec_input.shape)

        # Loop until the max_length is reached for the target lang (ENGLISH)
        state = decoder_initial_state
        for t in range(self.max_length_output):
            # print(dec_input.shape)
            predictions, state = self.decoder.one_step(dec_input, state)
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
            dec_input = tf.reshape([predicted_id], [-1,])

        return result# , sentence# , attention_plot

    def translate(self, sentence):
        result = self.evaluate_greedy(sentence)
        # print(result)
        # result = self.targ_lang.sequences_to_texts(result)
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))
        return result

nmt = NMT(checkpoint_dir='./training_pure_checkpoints',
          num_examples=57313,
          epochs=20
         )
nmt.build_dataset('./chn-eng.tsv')
nmt.build_model()
nmt.restore_weights()

nmt.translate("How old are you ?")
nmt.train()
nmt.translate("How old are you ?")
