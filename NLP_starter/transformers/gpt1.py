#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import tensorflow as tf
import numpy as np

from .transformer import EncoderLayer
from ..utils.normalization import LayerNormalization

def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    x: float Tensor to perform activation.

    Returns:
    `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    # config = {
#   "directionality": "bidi",
#   "hidden_act": "gelu",

#   "pooler_fc_size": 768,
#   "pooler_num_attention_heads": 12,
#   "pooler_num_fc_layers": 3,
#   "pooler_size_per_head": 128,
#   "pooler_type": "first_token_transform",
#   "type_vocab_size": 2,

    """
    def __init__(self,
                vocab_size=21128,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,  ## dff
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02,
                max_seq_len=128):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.max_seq_len = max_seq_len

class GPT1Decoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(GPT1Decoder, self).__init__(**kwargs)
        self.decoder_layers = [
            EncoderLayer(n_head=config.num_attention_heads,
                         d_model=config.hidden_size,
                         dff=config.intermediate_size,
                         dropout=config.hidden_dropout_prob,
                         layer_norm_epsilon=1e-12,
                         attention_use_scale=True,
                         attention_scale_factor=1 / math.sqrt(
                             config.hidden_size // config.num_attention_heads),
                         use_query_mask=False,
                         ff_activation=gelu,
                         causal=True
                        ) \
                for _ in range(config.num_hidden_layers)]

    def build(self, input_shape):
        for layer_module in self.decoder_layers:
            layer_module.build(input_shape)
        super(GPT1Decoder, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        """
        inputs:
            (batch, Te, emb_size)
        return:
            a list of tensor (batch, Te, d_model)
        """
        all_encoder_layers_list = []
        prev_output = inputs
        for layer_module in self.decoder_layers:
            prev_output = layer_module(inputs=prev_output, mask=mask,
                                       training=training)
            all_encoder_layers_list.append(prev_output)
        return all_encoder_layers_list


class GPT1Model(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(GPT1Model, self).__init__(**kwargs)
        self.config = config
        self.word_embeddings = tf.keras.layers.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.decoder = GPT1Decoder(self.config)

    def build(self, input_shape):
        self.word_embeddings.build(input_shape)
        self.position_embeddings.build(input_shape)
        self.decoder.build(input_shape + (self.config.hidden_size,))
        super(GPT1Model, self).build(input_shape)

    def call(self, inputs, mask, training=None):
        embedding_output = self.word_embeddings(inputs) + \
            self.position_embeddings(tf.range(self.config.max_seq_len))
        all_decoder_layers = self.decoder(embedding_output, mask=mask,
                                          training=training)
        sequence_output = all_decoder_layers[-1]

        return sequence_output


class PretrainGPT1Model(object):
    def __init__(self, config, init_checkpoint=None,
                 lm_coef = 0.5,
                 layer_norm_epsilon=1e-12,):
        self.config = config
        self.config.layer_norm_epsilon = layer_norm_epsilon
        self.config.lm_coef = 0.5
        self.basic_model = GPT1Model(config)
        self.basic_model.build((None, config.max_seq_len))
        # if init_checkpoint:
        #     self.basic_model.restore_weights(init_checkpoint)

    def build_pretrain_model(self, CLS_index, training=None):
        # inputs
        inputs = tf.keras.layers.Input((self.config.max_seq_len,), dtype=tf.int64)
        # input_mask 并没有输入到gpt模型中，只生效在最后到lm_loss上
        input_mask = tf.keras.layers.Input((self.config.max_seq_len,), dtype=tf.float32)
        cls_label = tf.keras.layers.Input((1,))
        # weights
        output_weights = self.basic_model.word_embeddings.embeddings
        clf_dense = tf.keras.layers.Dense(1)
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # gpt-1 model, no mask
        h = self.basic_model(inputs, mask=None, training=training)

        # classifier loss
        pool_ids = tf.argmax(tf.cast(tf.equal(inputs, CLS_index), tf.int32), axis=1)
        # index = list(zip(range(tf.shape(pool_ids)[0]), pool_ids))
        index = tf.stack([tf.range(tf.shape(pool_ids)[0], dtype=tf.int64), pool_ids], axis=1)

        cls_h = tf.gather_nd(h, index)
        logits = clf_dense(cls_h)
        clf_losses = bce_loss(cls_label, logits)

        # masked lm loss
        lm_h = tf.reshape(h[:, :-1], [-1, self.config.hidden_size])
        lm_logits = tf.matmul(lm_h, output_weights, transpose_b=True)
        lm_labels = tf.reshape(inputs[:, 1:], [-1])
        # lm_loss = scce_loss(lm_labels, lm_logits) # 直接用scce无法编译，改成自己实现。
        # per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        per_example_loss = -tf.reduce_sum(
            tf.nn.log_softmax(lm_logits, axis=-1) * \
            tf.one_hot(lm_labels, depth=self.config.vocab_size), axis=1)
        # (batch, max_seq_len -1)
        per_example_loss = tf.reshape(per_example_loss,
                                      (-1, self.config.max_seq_len - 1))
        mask = input_mask[:, 1:] # 最后一token预测到是0，再往后就是0预测0，没有意义，mask掉。
        lm_loss = tf.reduce_sum(per_example_loss * mask, axis=1) / tf.reduce_sum(mask, axis=1)
        print('debug-lm_loss', lm_loss.shape)
        lm_loss = tf.reduce_mean(lm_loss)
        total_loss = clf_losses + self.config.lm_coef * lm_loss

        self.model = tf.keras.models.Model(inputs=[inputs, input_mask, cls_label], outputs=total_loss)

        def fake_loss(y_true, y_pred):
            return y_pred
        self.model.compile('Adam', loss=fake_loss, metrics=[fake_loss])

    def train(self, dataset):
        self.model.fit(x=dataset, epochs=2)



