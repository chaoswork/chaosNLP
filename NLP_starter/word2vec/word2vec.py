#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import tensorflow.keras.backend as backend
from tensorflow.python.ops import control_flow_ops


from collections import defaultdict
import numpy as np
import tqdm
import math
import random


class SaveEmbPerEpoch(tf.keras.callbacks.Callback):
    def set_emb(self, emb, vocab, inverse_vocab,  word_embedding_file):
        self.word_embedding_file = word_embedding_file
        self.words_embedding_in = emb
        self.vocabulary = vocab
        self.inverse_vocab = inverse_vocab

    def on_epoch_end(self, epoch, logs=None):
        with open(self.word_embedding_file + '.{}'.format(epoch), 'w') as f:
            weights = self.words_embedding_in.get_weights()[0]
            for i in range(len(self.vocabulary)):
                emb = weights[i,:]
                line = '{} {}\n'.format(
                    self.inverse_vocab[i],
                    ' '.join([str(x) for x in emb])
                )
                f.write(line)


# wget http://mattmahoney.net/dc/text8.zip -O text8.gz
# gzip -d text8.gz -f
train_file = './text8'

class Word2Vec(object):
    def __init__(self, train_file, sample=1e-4, embedding_dim=200):
        self.train_file = train_file
        self.MAX_SENTENCE_LENGTH = 1024
        self.min_count = 5
        self.subsampling_power = 0.75
        self.sample = sample
        self.save_vocab = './vocab.txt'
        self.window = 5
        self.negative = 5
        self.skip_gram_by_src = True
        self.embedding_dim = embedding_dim
        self.word_embedding_file = 'word_embedding.txt'
        self.tfrecord_file = 'word_pairs.tfrecord'

        self.vocabulary = None
        self.next_random = 1
        self.table_size = 10 ** 8
        self.batch_size = 256
        self.epochs = 5
        self.gen_tfrecord = True

        # build vocabulary
        print('build vocabulary ...')
        self.build_vocabulary()
        # build dataset
        print('transfer data to tfrecord ...')
        if self.gen_tfrecord:
            self.data_to_tfrecord()
        # 使用from_generator，速度非常慢，遍历100个句子需要50s
        # self.dataset = tf.data.Dataset.from_generator(
        #     self.train_data_generator,
        #     output_types=(tf.int32, tf.int32),
        #     output_shapes=((2,), (),)
        # ).batch(self.batch_size).prefetch(1)
        # 使用tfrecord后，100个句子需要6s
        self.dataset = self.make_dataset()

        # build model
        print('build model ...')
        self.build_model()

    def make_dataset(self):
        def parse_tfrecord(record):
            features = tf.io.parse_single_example(
                record,
                features={
                    'pair': tf.io.FixedLenFeature([2], tf.int64),
                    'label': tf.io.FixedLenFeature([1], tf.float32)
                })
            label = features['label']
            pair = features['pair']
            return pair, label
        dataset = tf.data.TFRecordDataset(self.tfrecord_file)\
            .map(parse_tfrecord, num_parallel_calls=8)\
            .batch(self.batch_size).prefetch(self.batch_size)
        return dataset

    def build_unigram_table(self, word_prob):
        self.table = [0] * self.table_size
        word_index = 1
        cur_length = word_prob[word_index]
        for a in range(len(self.table)):
            self.table[a] = word_index
            if a / len(self.table) > cur_length:
                word_index += 1
                cur_length += word_prob[word_index]
            if word_index >= len(self.vocabulary):
                word_index -= 1


    def build_vocabulary(self):
        word_freqs = defaultdict(int)

        for tokens in self.data_generator():
            for token in tokens:
                word_freqs[token] += 1

        word_freqs = {word: freq for word, freq in word_freqs.items() \
                          if freq >= self.min_count}

        self.vocabulary = {word: index + 1 for (index, (word, freq)) in enumerate(
            sorted(word_freqs.items(), key=lambda x: x[1], reverse=True))}
        self.vocabulary['</s>'] = 0
        self.inverse_vocab = {index: token for token, index in self.vocabulary.items()}
        # save vocab
        with open(self.save_vocab, 'w') as f:
            for i in range(len(self.vocabulary)):
                word = self.inverse_vocab[i]
                if i > 0:
                    freq = word_freqs[word]
                else:
                    freq = 0
                f.write(f"{word} {freq}\n")


        # 负采样的采样概率，f(w)^(3/4)/Z
        train_words_ns = sum([freq**(self.subsampling_power) for freq in word_freqs.values()])
        self.ns_word_prob = {self.vocabulary[word]: (freq**self.subsampling_power) / train_words_ns for word, freq in word_freqs.items()}
        self.build_unigram_table(self.ns_word_prob)
#         self.unigrams_prob = [0]
#         for i in range(1, len(self.vocabulary)):
#             # print(inverse_vocab[i])
#             self.unigrams_prob.append(self.ns_word_prob[i])

        # (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
        # subsampling
        if self.sample > 0:
            train_words = sum([freq for freq in word_freqs.values()])
            self.subsampling_drop_ratio = {
                word: (math.sqrt(freq / (self.sample * train_words)) + 1) * (self.sample * train_words) / freq \
                    for word, freq in word_freqs.items()
            }

    def build_model(self):
        vocab_size = len(self.vocabulary)
        #embedding_dim = 100
        inputs = Input(shape=(2,))
        target = inputs[:, 0:1]
        context = inputs[:, 1:2]
        self.words_embedding_in = tf.keras.layers.Embedding(
                    vocab_size,
                    self.embedding_dim,
                    input_length=1,
                    name="word_embedding_in"
                )
        self.words_embedding_out = tf.keras.layers.Embedding(
                    vocab_size,
                    self.embedding_dim,
                    input_length=1,
                    name="word_embedding_out"
                )
        word_emb = self.words_embedding_in(target)
        context_emb = self.words_embedding_out(context)
        dots = tf.keras.layers.Dot(axes=(2, 2))([word_emb, context_emb])
        outputs = tf.keras.layers.Flatten()(dots)
        self.model = Model(inputs, outputs)


        self.model.compile(
            optimizer='adam',
            #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            # loss=tf.keras.losses.binary_crossentropy(from_logits=True),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])

    def train(self):
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        my_callback = SaveEmbPerEpoch()
        my_callback.set_emb(self.words_embedding_in,
                            self.vocabulary, self.inverse_vocab,
                            self.word_embedding_file)
        self.model.fit(word2vec.dataset, epochs=self.epochs, callbacks=[my_callback])

    def save_word_embeddings(self):
        with open(self.word_embedding_file, 'w') as f:
            f.write('{} {}\n'.format(len(self.vocabulary), self.embedding_dim))
            weights = self.words_embedding_in.get_weights()[0]
            for i in range(len(self.vocabulary)):
                emb = weights[i, :]
                line = '{} {}\n'.format(
                    self.inverse_vocab[i],
                    ' '.join([str(x) for x in emb])
                )
                f.write(line)

    def data_to_tfrecord(self):
        with tf.io.TFRecordWriter(self.tfrecord_file) as writer:
            for item in tqdm.tqdm(self.train_data_generator()):
                pair, label = item
                feature = {
                    'pair': tf.train.Feature(int64_list=tf.train.Int64List(value=pair)),
                    'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature)) 
                writer.write(example.SerializeToString()) 

    def train_data_generator(self):
        cnt = 0
        for tokens_index in self.tokens_generator():
            # print(len(tokens_index), cnt)

            for i in range(len(tokens_index)):
                # print('cnt={}, i={}'.format(cnt, i))
                cnt += 1
                word = tokens_index[i]
                b = random.randint(0, self.window - 1)
                window_t = self.window - b
                # c为上下文坐标
#                 context_ = [tokens_index[c] for c in range(i - window_t, i + window_t + 1) \
#                                if c >=0 and c <=len(tokens_index) and c != i]
#                 print('window_t = {}, contexts words={}'.format(window_t, context_))
                for c in range(i - window_t, i + window_t + 1):
                    # 越界的和中心词跳过。
                    if c < 0 or c >= len(tokens_index) or c == i:
                        continue

                    context_word = tokens_index[c]
                    # print('c={}, context_word={}'.format(c, context_word))

                    # 构造副样本
                    # 采用np.random.choice的方法，10句话要5分钟。
                    # 采用tf.random.fixed_unigram_candidate_sampler，10句话要7分钟。
                    # 所以最后还得用hash的方法搞。10句话基本不需要时间
                    # 但是改成dataset后，仍然需要5s
#                     neg_indexs = [np.random.choice(
#                         list(self.ns_word_prob.keys()),
#                         p=list(self.ns_word_prob.values())) for _ in range(self.negative)]
                    #
                    neg_indexs = [self.table[random.randint(0, len(self.table) - 1)] \
                                      for _ in range(self.negative)]

                    if self.skip_gram_by_src:
                        yield [context_word, word], 1.0
                        for negative_word in neg_indexs:
                            if negative_word != word:
                                yield [context_word, negative_word], 0.0
                    else:
                        yield [word, context_word], 1.0
                        for negative_word in neg_indexs:
                            if negative_word != word:
                                yield [word, negative_word], 0.0


    def tokens_generator(self):
        cnt = 0
        for tokens in self.data_generator():
            tokens_index = []
            for token in tokens:
                if token not in self.vocabulary:
                    continue
                if self.sample > 0:
                    if self.subsampling_drop_ratio[token] < np.random.uniform(0, 1):
                    # if self.subsampling_drop_ratio[token] < self.w2v_random():
                        continue
                tokens_index.append(self.vocabulary[token])
            # if cnt == 10:
            #     return None
            cnt += 1
            yield tokens_index

    def data_generator_from_memery(self):
        data = open(train_file).readlines()[0].split(' ')
        cur_tokens = []
        index = 0
        while index + 100 < len(data):
            yield data[index: index + 100]
            index += 100
        yield data[index: ]



        # for i in range(len(data)):
        #     cur_tokens.append(data[i])
        #     if i % 100 == 0:
        #         yield cur_tokens
        #         cur_tokens = []


    def data_generator(self):
        prev = ''
        with open(train_file) as f:
            while True:
                buffer = f.read(self.MAX_SENTENCE_LENGTH)
                if not buffer:
                    break
                # print('|{}|'.format(buffer))
                lines = (prev + buffer).split('\n')
                # print(len(lines))
                for idx, line in enumerate(lines):
                    tokens = line.split(' ')
                    if idx == len(lines) - 1:
                        cur_tokens = [x for x in tokens[:-1] if x]
                        prev = tokens[-1]
                    else:
                        cur_tokens = [x for x in tokens if x]


                    yield cur_tokens



word2vec = Word2Vec(train_file, sample=1e-4)

word2vec.train()

