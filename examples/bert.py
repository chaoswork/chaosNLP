#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
import tensorflow as tf

from NLP_starter.seq2seq.bert import BertConfig
from NLP_starter.seq2seq.bert import BertModel

config = BertConfig(type_vocab_size=2, max_seq_len=20)

model = BertModel(config)
model.build([(None, config.max_seq_len), (None, config.max_seq_len), (None, config.max_seq_len)])
model.restore_weights(None)
token_ids = [101, 276, 102]
token_ids = [101, 276, 245, 235, 984, 234, 102]
mask = [1] * len(token_ids)
input_ids = tf.pad([token_ids], [[0, 0], [0, config.max_seq_len - len(token_ids)]])
mask_ids = tf.pad([mask], [[0, 0], [0, config.max_seq_len - len(token_ids)]])
mask_ids = tf.cast(mask_ids, tf.bool)
input_type_ids = tf.constant([0] * config.max_seq_len)
position_ids = tf.constant(list(range(config.max_seq_len)))

a = model([input_ids, input_type_ids, position_ids], mask=mask_ids)
print(a)
