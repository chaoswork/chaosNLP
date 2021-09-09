#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-3,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer

    def build(self, input_shape):
#         if not isinstance(input_shape, list):
#             input_shape = [input_shape]
#         if self.axis < 0:
#             self.axis += len(input_shape)

        self.gamma = self.add_weight(
            name='gamma',
            shape=(input_shape[-1],),
            initializer=self.gamma_initializer
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(input_shape[-1],),
            initializer=self.beta_initializer
        )
        super(LayerNormalization, self).build(input_shape)
    def call(self, inputs, training=None):

        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=self.axis, keepdims=True)

        y = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return y * self.gamma + self.beta


