#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
import tensorflow as tf
import tensorflow_text as text

from NLP_starter.transformers.bert import BertConfig
from NLP_starter.transformers.gpt1 import PretrainGPT1Model


config = BertConfig(vocab_size=30522, max_seq_len=128)


batch_size = 12
test_file_path = './data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
train_file_path = './data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv'

en_tokenizer = text.BertTokenizer(
    './data/uncased_L-12_H-768_A-12/vocab.txt',
    lower_case=True,
    # token_out_type=tf.string,
    preserve_unused_token=True
)


def get_vocab_index(tokenizer, word):
    return tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(
        tf.constant(word)).numpy()


START = '[unused0]'
START_index = get_vocab_index(en_tokenizer, START)
SEP_index = get_vocab_index(en_tokenizer, '[SEP]')
CLS_index = get_vocab_index(en_tokenizer, '[CLS]')
print(START_index, SEP_index, CLS_index)


def get_dataset(file_path):
    LABEL_COLUMN = 'AnswerRightEnding'
    dataset = tf.data.experimental.make_csv_dataset(
          file_path,
          batch_size=batch_size,
          label_name=LABEL_COLUMN,
          na_value="?",
          num_epochs=1,
          ignore_errors=True)
    return dataset


def parse_data(*raw_data):
    example, labels = raw_data
    token_a = tf.strings.join([example['InputSentence1'],
                               example['InputSentence2'],
                               example['InputSentence3'],
                               example['InputSentence4'],
                              ], separator=' ')
    token_a = en_tokenizer.tokenize(token_a).merge_dims(-2, -1)
    token_b = en_tokenizer.tokenize(example['RandomFifthSentenceQuiz1']).merge_dims(-2, -1)
    token_c = en_tokenizer.tokenize(example['RandomFifthSentenceQuiz2']).merge_dims(-2, -1)
    max_len = config.max_seq_len // 2 - 2

    print(token_a.shape)
    tile_size = tf.shape(labels)[0]
    start_column =  tf.tile(tf.constant([[START_index]], dtype=tf.int64),
                            [tile_size, 1])
    sep_column =  tf.tile(tf.constant([[SEP_index]], dtype=tf.int64),
                            [tile_size, 1])
    cls_column =  tf.tile(tf.constant([[CLS_index]], dtype=tf.int64),
                            [tile_size, 1])
#     print(tf.tile([[START_index]], [tf.shape(token_a)[0], 1]))
    pair_ab = tf.concat([start_column,
                        token_a[:,:max_len],
                        sep_column,
                        token_b[:,:max_len], cls_column], axis=1).to_tensor()

    pair_ac = tf.concat([start_column,
                        token_a[:,:max_len],
                        sep_column,
                        token_c[:,:max_len], cls_column], axis=1).to_tensor()

    pair_ab = tf.pad(pair_ab, [[0, 0], [0, config.max_seq_len - tf.shape(pair_ab)[1]]])
    pair_ac = tf.pad(pair_ac, [[0, 0], [0, config.max_seq_len - tf.shape(pair_ac)[1]]])
    example = tf.concat([pair_ab, pair_ac], axis=0)    # xmb and X
    mask = tf.cast(tf.greater(example, 0), tf.float32) # mmb and M

    labels = tf.one_hot(labels - 1, depth=2)
    labels = tf.concat([labels[:,0], labels[:,1]], axis=0) # Y

    return (example, mask, labels), tf.zeros((1,))


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

raw_train_data = raw_train_data.map(parse_data)
raw_test_data = raw_test_data.map(parse_data)

model = PretrainGPT1Model(config)
model.build_pretrain_model(CLS_index, training=True)
model.train(raw_train_data)
