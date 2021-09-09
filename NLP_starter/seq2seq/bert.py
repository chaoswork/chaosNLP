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



class BertEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, layer_norm_epsilon=1e-12, **kwargs):
        super(BertEmbeddings, self).__init__(**kwargs)
        self.config = config
        self.word_embeddings = tf.keras.layers.Embedding(
            config.vocab_size, config.hidden_size,
            name="word_embeddings")
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size, config.hidden_size,
            name="token_type_embeddings")
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings, config.hidden_size,
            name="position_embeddings")


        # tensorflow的源代码中variance_epsilon=1e-12写死了。
        # /anaconda2/lib/python2.7/site-packages/tensorflow/contrib/layers/python/layers/layers.py
        # self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.layer_norm = LayerNormalization(epsilon=layer_norm_epsilon)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape_list):
        input_shape = input_shape_list[0]
        self.token_type_embeddings.build(input_shape)
        self.position_embeddings.build(input_shape)
        self.word_embeddings.build(input_shape)
        # self.layer_norm.build(self.config.hidden_size)
        self.layer_norm.build(input_shape + (self.config.hidden_size, ))
        self.dropout.build(self.config.hidden_size)
        super(BertEmbeddings, self).build(input_shape_list)

    def call(self, inputs):
        input_ids, token_type_ids, position_ids = inputs
        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(BertEncoder, self).__init__(**kwargs)
        self.encoder_layers = [
            EncoderLayer(n_head=config.num_attention_heads,
                         d_model=config.hidden_size,
                         dff=config.intermediate_size,
                         dropout=config.hidden_dropout_prob,
                         layer_norm_epsilon=1e-12,
                         attention_use_scale=True,
                         attention_scale_factor=1 / math.sqrt(
                             config.hidden_size // config.num_attention_heads),
                         use_query_mask=False,
                         ff_activation=gelu

                        ) \
                for _ in range(config.num_hidden_layers)]

    def build(self, input_shape):
        for layer_module in self.encoder_layers:
            layer_module.build(input_shape)
        super(BertEncoder, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        inputs:
            (batch, Te, emb_size)
        return:
            a list of tensor (batch, Te, d_model)
        """
        all_encoder_layers_list = []
        prev_output = inputs
        for layer_module in self.encoder_layers:
            prev_output = layer_module(inputs=prev_output, mask=mask)
            all_encoder_layers_list.append(prev_output)
        return all_encoder_layers_list

class BertPooler(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(BertPooler, self).__init__(**kwargs)
        self.pooled_dense = tf.keras.layers.Dense(
            config.hidden_size, activation='tanh')

    def build(self, input_shape):
        self.pooled_dense.build(input_shape)
        super(BertPooler, self).build(input_shape)

    def call(self, inputs):
        """
        inputs:
            all encoder layer list
        return:
            (batch, hidden_size)
        """
        sequence_output = inputs[-1]
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = self.pooled_dense(first_token_tensor)
        return pooled_output

class BertModel(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(BertModel, self).__init__(**kwargs)
        self.config = config
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

    def build(self, input_shape):
        self.embeddings.build(input_shape)
        self.encoder.build(input_shape[0] + (self.config.hidden_size,))
        self.pooler.build(input_shape[0] + (self.config.hidden_size,))
        super(BertModel, self).build(input_shape)

    def call(self, inputs, mask):
        # input_ids, token_type_ids, position_ids = inputs
        embedding_output = self.embeddings(inputs)
        all_encoder_layers = self.encoder(embedding_output, mask=mask)
        pooled_output = self.pooler(all_encoder_layers)

        return pooled_output

    def restore_weights(self, checkpoint_file):
        """
        加载模型
        """
        model_path = '/home/odin/chaohuang/data/chinese_L-12_H-768_A-12/'
        #tf.compat.v1.train.import_meta_graph(os.path.join(model_path))
        loaded = tf.train.load_checkpoint(os.path.join(model_path, 'bert_model.ckpt'))
        var2shape_map = loaded.get_variable_to_shape_map()
#         for name in var2shape_map:
#             print(name, loaded.get_tensor(name).shape)
        names = []
        for name in var2shape_map:
            name_tree = name.split('/')
            pre_trained_value = loaded.get_tensor(name)

            if 'embeddings' in name_tree:
                layer_norm_weights = {}
                if name_tree[-1].endswith('embeddings'):
                    getattr(self.embeddings, name_tree[-1]).set_weights([pre_trained_value])
                else:
                    tf.keras.backend.set_value(
                        getattr(self.embeddings.layer_norm, name_tree[-1]), pre_trained_value)
                # print(name, pre_trained_value.shape)
            # set encoder weights
            elif name.startswith('bert/encoder/layer'):
                layer_index = int(name_tree[2].split('_')[-1])
                cur_encoder_layer = self.encoder.encoder_layers[layer_index]
                if name_tree[3] == 'attention':
                    attention = cur_encoder_layer.multi_head_attention
                    layer_norm_attention = cur_encoder_layer.layer_norm_attention
                    # attention_out_dense = cur_encoder_layer.attention_out_dense
                    if name_tree[4] == 'self':
                        map_dict = {
                            'query': attention.WQ,
                            'key': attention.WK,
                            'value': attention.WV
                        }
                        tf.keras.backend.set_value(
                            getattr(map_dict[name_tree[5]], name_tree[6]),
                            pre_trained_value
                        )
                    elif name_tree[4] == 'output':
                        map_dict = {
                            'dense': attention.Wo,
                            'LayerNorm': layer_norm_attention,

                        }
                        tf.keras.backend.set_value(
                            getattr(map_dict[name_tree[5]], name_tree[6]),
                            pre_trained_value
                        )
                elif name_tree[3] == 'intermediate':
                    tf.keras.backend.set_value(
                            getattr(cur_encoder_layer.dense1, name_tree[5]),
                            pre_trained_value
                        )
                elif name_tree[3] == 'output':
                    map_dict = {
                            'LayerNorm': cur_encoder_layer.layer_norm_dense,
                            'dense': cur_encoder_layer.dense2
                        }
                    tf.keras.backend.set_value(
                            getattr(map_dict[name_tree[4]], name_tree[5]),
                            pre_trained_value
                        )


            elif 'pooler' in name_tree:
                tf.keras.backend.set_value(
                    getattr(self.pooler.pooled_dense, name_tree[3]),
                            pre_trained_value
                        )
            else:
                print(name, pre_trained_value.shape)


