#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
import tensorflow as tf

from NLP_starter.seq2seq.bert import BertConfig
from NLP_starter.seq2seq.bert import BertModel
from NLP_starter.seq2seq.bert import PretrainBertModel

# config = BertConfig(type_vocab_size=2, max_seq_len=20)
# 
# model = BertModel(config)
# model.build([(None, config.max_seq_len), (None, config.max_seq_len), (None, config.max_seq_len)])
# model.restore_weights(None)
# token_ids = [101, 276, 102]
# token_ids = [101, 276, 245, 235, 984, 234, 102]
# mask = [1] * len(token_ids)
# input_ids = tf.pad([token_ids], [[0, 0], [0, config.max_seq_len - len(token_ids)]])
# mask_ids = tf.pad([mask], [[0, 0], [0, config.max_seq_len - len(token_ids)]])
# mask_ids = tf.cast(mask_ids, tf.bool)
# input_type_ids = tf.constant([0] * config.max_seq_len)
# position_ids = tf.constant(list(range(config.max_seq_len)))
# 
# a = model([input_ids, input_type_ids, position_ids], mask=mask_ids)

config = BertConfig(type_vocab_size=2) #, max_seq_len=20)

pbm = PretrainBertModel(config)
print(pbm.build_pretrain_model())



def decode_record(record, config):
    max_seq_length = config.max_seq_len
    max_predictions_per_seq = config.max_predictions_per_seq
    name_to_features = {
            "input_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.io.FixedLenFeature([1], tf.int64),
        }
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
          t = tf.cast(t, tf.int32)
        example[name] = t
    print(example)
    # inputs = [input_ids, input_mask, segment_ids, masked_lm_positions,
    #               masked_lm_ids, masked_lm_weights, next_sentence_labels]
    inputs = (example["input_ids"], example["input_mask"],
              example["segment_ids"], example["masked_lm_positions"],
              example["masked_lm_ids"],
              example["masked_lm_weights"],
              example["next_sentence_labels"])
    # return inputs, tf.ones((1,))
    return example, tf.ones((1,))

dataset = tf.data.TFRecordDataset('/home/odin/chaohuang/git-local/nlp/bert/tf_examples.tfrecord')

# for raw_record in d.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)
dataset = dataset.map(lambda x: decode_record(x, config)).batch(1)

# print(next(iter(dataset)))
pbm.train(dataset)
