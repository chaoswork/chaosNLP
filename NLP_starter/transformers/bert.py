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

    def call(self, inputs, training=None):
        input_ids, token_type_ids, position_ids = inputs
        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training)
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
                         ff_activation=gelu) for _ in range(config.num_hidden_layers)]

    def build(self, input_shape):
        for layer_module in self.encoder_layers:
            layer_module.build(input_shape)
        super(BertEncoder, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        """
        inputs:
            (batch, Te, emb_size)
        return:
            a list of tensor (batch, Te, d_model)
        """
        all_encoder_layers_list = []
        prev_output = inputs
        for layer_module in self.encoder_layers:
            prev_output = layer_module(inputs=prev_output, mask=mask, training=training)
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

    def call(self, inputs, mask, training=None):
        # input_ids, token_type_ids, position_ids = inputs
        embedding_output = self.embeddings(inputs, training=training)
        all_encoder_layers = self.encoder(embedding_output, mask=mask, training=training)
        self.sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(all_encoder_layers)

        return pooled_output

    def restore_weights(self, checkpoint_file):
        """
        加载模型
        """
        model_path = './data/chinese_L-12_H-768_A-12/'
        # tf.compat.v1.train.import_meta_graph(os.path.join(model_path))
        loaded = tf.train.load_checkpoint(os.path.join(model_path, 'bert_model.ckpt'))
        var2shape_map = loaded.get_variable_to_shape_map()
#         for name in var2shape_map:
#             print(name, loaded.get_tensor(name).shape)
        for name in var2shape_map:
            name_tree = name.split('/')
            pre_trained_value = loaded.get_tensor(name)

            if 'embeddings' in name_tree:
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
                        pre_trained_value)
                elif name_tree[3] == 'output':
                    map_dict = {
                        'LayerNorm': cur_encoder_layer.layer_norm_dense,
                        'dense': cur_encoder_layer.dense2 }
                    tf.keras.backend.set_value(
                        getattr(map_dict[name_tree[4]], name_tree[5]),
                        pre_trained_value)

            elif 'pooler' in name_tree:
                tf.keras.backend.set_value(
                    getattr(self.pooler.pooled_dense, name_tree[3]),
                            pre_trained_value
                        )
            else:
                print(name, pre_trained_value.shape)



class PretrainBertModel(object):
    def __init__(self, config, init_checkpoint=None,
                 max_predictions_per_seq=20,
                 layer_norm_epsilon=1e-12,):
        self.config = config
        self.config.layer_norm_epsilon = layer_norm_epsilon
        self.config.max_predictions_per_seq = max_predictions_per_seq
        self.basic_model = BertModel(config)
        self.basic_model.build([(None, config.max_seq_len), (None, config.max_seq_len), (None, config.max_seq_len)])
        if init_checkpoint:
            self.basic_model.restore_weights(init_checkpoint)

    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        batch_size, seq_length, width = tf.shape(sequence_tensor)
        flat_offsets = tf.reshape(
          tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                        [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor

    def build_pretrain_model(self, training=None):
        # inputs
        input_ids = tf.keras.layers.Input(
            (self.config.max_seq_len,), dtype=tf.int32, name="input_ids")
        input_mask = tf.keras.layers.Input(
            (self.config.max_seq_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(
            (self.config.max_seq_len,), dtype=tf.int32, name="segment_ids")
        masked_lm_positions = tf.keras.layers.Input(
            (self.config.max_predictions_per_seq,), dtype=tf.int32,
            name="masked_lm_positions")
        masked_lm_ids = tf.keras.layers.Input(
            (self.config.max_predictions_per_seq,), dtype=tf.int32,
            name="masked_lm_ids")
        masked_lm_weights = tf.keras.layers.Input(
            (self.config.max_predictions_per_seq,), dtype=tf.float32,
            name="masked_lm_weights")
        next_sentence_labels = tf.keras.layers.Input(
            (1,), dtype=tf.int32, name="next_sentence_labels")

        # weights
        self.cls_pred_tsfm_dense = tf.keras.layers.Dense(self.config.hidden_size)
        self.cls_pred_tsfm_dense.build((None, self.config.hidden_size))
        self.cls_pred_tsfm_layer_norm = LayerNormalization(
            epsilon=self.config.layer_norm_epsilon)
        self.cls_pred_tsfm_layer_norm.build((None, self.config.hidden_size))
        self.output_bias = tf.Variable(
            tf.zeros((self.config.vocab_size,)))
        self.seq_relationship_dense = tf.keras.layers.Dense(2)
        self.seq_relationship_dense.build((None, self.config.hidden_size))

        # build model
        position_ids = tf.constant(list(range(self.config.max_seq_len)))

        pooled_output = self.basic_model(
            [input_ids, segment_ids, position_ids],
            mask=input_mask, training=training)
        sequence_output = self.basic_model.sequence_output

        # masked lm log probs
        input_tensor = self.gather_indexes(sequence_output, masked_lm_positions)
        output_weights = self.basic_model.embeddings.word_embeddings.embeddings

        input_tensor = self.cls_pred_tsfm_dense(input_tensor)
        input_tensor = self.cls_pred_tsfm_layer_norm(input_tensor)
        masked_lm_logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        masked_lm_logits = tf.nn.bias_add(masked_lm_logits, self.output_bias)
        masked_lm_log_probs = tf.nn.log_softmax(masked_lm_logits, axis=-1)

        # next sentence log probs
        next_sentence_logits = self.seq_relationship_dense(pooled_output)
        next_sentence_log_probs = tf.nn.log_softmax(next_sentence_logits, axis=-1)

        # masked lm loss
        label_ids = tf.reshape(masked_lm_ids, [-1])
        label_weights = tf.reshape(masked_lm_weights, [-1])
        one_hot_labels = tf.one_hot(
            label_ids, depth=self.config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(masked_lm_log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        masked_lm_loss = numerator / denominator

        # next sentense
        ns_labels = tf.reshape(next_sentence_labels, [-1])
        one_hot_labels = tf.one_hot(ns_labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * next_sentence_log_probs, axis=-1)
        next_sentence_loss = tf.reduce_mean(per_example_loss)

        total_loss = masked_lm_loss + next_sentence_loss

        # inputs = [input_ids, input_mask, segment_ids, masked_lm_positions,
        #           masked_lm_ids, masked_lm_weights, next_sentence_labels]
        # labels = [masked_lm_ids, masked_lm_weights, next_sentence_labels]
        # outputs = [masked_lm_log_probs, next_sentence_log_probs]

        def bert_loss(y_true, y_pred):
            return y_pred

        self.model = tf.keras.models.Model(
            inputs=[input_ids, input_mask, segment_ids, masked_lm_positions,
                    masked_lm_ids, masked_lm_weights, next_sentence_labels],
            # outputs=next_sentence_loss)
            outputs=total_loss)
        self.model.compile('Adam',
                           loss=bert_loss,
                           metrics=[bert_loss])
        return self.model.summary()

    def train(self, dataset):
        self.model.fit(x=dataset, epochs=2)



