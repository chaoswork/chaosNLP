#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np



def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(50, 512)
print (pos_encoding.shape)


def _lower_triangular_mask(shape):
    """
    左下角全为True的Tensor
    """
    row_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return tf.greater_equal(row_index, col_index)


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, use_scale=False, causal=False, **kwargs):
        self.use_scale = use_scale
        self.causal = causal
        self.scale = 1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0][:-1] + input_shape[1][-1:]
        return input_shape

    def call(self, inputs, mask=None, **kwargs):
        """
        inputs: list or tensor
            inputs is a list like [query, value, key]
            or just one tensor stands for self-attention
            query: [batch_size, Tq, dim_k]
            value: [batch_size, Tv, dim_v]
            key: [batch_size, Tv, dim_k]
        mask: a list, [query_mask, value_mask]
            query_mask shape: [batch_size, Tq]
            value_mask shape: [batch_size, Tv]
        Return:
            [batch_size, Tq, dim_v]
        """
        if isinstance(inputs, list):
            query, value, key = inputs
        else:
            query = value = key = inputs

        if mask:
            query_mask, value_mask = mask
            if query_mask is not None:
                query_mask = tf.expand_dims(query_mask, -1)  # (batch, Tq, 1)
            if value_mask is not None:
                value_mask = tf.expand_dims(value_mask, -2)  # (batch, 1, Tv)
        else:
            query_mask, value_mask = None, None

        if self.use_scale:
            feature_dim = query.shape[-1]
            self.scale = 1 / tf.sqrt(tf.cast(feature_dim, tf.float32))
        # (batch, Tq, Tv)
        scores = tf.matmul(query, key, transpose_b=True) * self.scale
        if self.causal:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the future
            # into the past.¬
            # causal_mask_shape = (1, Tq, Tv)
            scores_shape = tf.shape(scores)
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0)
            causal_mask = _lower_triangular_mask(causal_mask_shape)

        else:
            causal_mask = None
        # merge causal_mask and value mask
        if causal_mask is None:
            scores_mask = value_mask
        elif value_mask is None:
            scores_mask = causal_mask
        else:
            # value_mask: (batch, 1, Tv)
            # causal_mask: (1, Tq, Tv)
            # scores_mask: (batch, Tq, Tv)
            scores_mask = tf.logical_and(value_mask, causal_mask)

        if scores_mask is not None:
            # 要mask的位置设置一个特别小的数，这样softmax结果为0
            padding_mask = tf.logical_not(scores_mask)
            scores -= 1.e9 * tf.cast(padding_mask, dtype=tf.float32)
        weight = tf.nn.softmax(scores)

        context = tf.matmul(weight, value)
        if query_mask is not None:
            context *= tf.cast(query_mask, context.dtype)
        return context

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n_head, d_model,
                 use_scale=False, causal=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_v = self.d_model // self.n_head
        self.use_scale = use_scale
        self.causal = causal

        self.WQ = tf.keras.layers.Dense(self.n_head * self.d_k)
        self.WK = tf.keras.layers.Dense(self.n_head * self.d_k)
        self.WV = tf.keras.layers.Dense(self.n_head * self.d_v)
        self.Wo = tf.keras.layers.Dense(self.d_model)
        # self.attention = tf.keras.layers.Attention(causal=self.causal)
        self.attention = ScaledDotProductAttention(
            use_scale=self.use_scale,
            causal=self.causal)

    def call(self, inputs, mask=None,
             multi_head_type='split_concat', training=None):
        """
        inputs: list or tensor
            inputs is a list like [query, value, key]
            or just one tensor stands for self-attention
            query: [batch_size, Tq, dim_k]
            value: [batch_size, Tv, dim_v]
            key: [batch_size, Tv, dim_k]
        mask: a list, [query_mask, value_mask]
            query_mask shape: [batch_size, Tq]
            value_mask shape: [batch_size, Tv]
        multi_head_type:
            Implemented using diffenent type
        Return:
            [batch_size, Tq, d_model]
        """
        if isinstance(inputs, list):
            query, value, key = inputs
        else:
            query = value = key = inputs

        query_linear_project = self.WQ(query) # (batch, Tq, n_head * d_k)
        value_linear_project = self.WV(value) # (batch, Tv, n_head * d_v)
        key_linear_project = self.WK(key)     # (batch, Tv, n_head * d_k)

        if multi_head_type == 'split_concat':
            query_n_heads = tf.concat(
                tf.split(query_linear_project, self.n_head, axis=-1), axis=0)
            value_n_heads = tf.concat(
                tf.split(value_linear_project, self.n_head, axis=-1), axis=0)
            key_n_heads = tf.concat(
                tf.split(key_linear_project, self.n_head, axis=-1), axis=0)
            tile_mask = None
            if mask:
                tile_mask = [tf.tile(m, [self.n_head, 1]) \
                                 if m is not None else None \
                                     for m in mask]
            # (batch * n_head, Tq, d_v)
            attention = self.attention(
                [query_n_heads, value_n_heads, key_n_heads],
                mask=tile_mask
            )
            # (batch, Tq, n_head * d_v)
            attention = tf.concat(tf.split(attention, self.n_head, 0), axis=-1)
        elif multi_head_type == 'transpose_reshape':
            Tq = query_linear_project.shape[1]
            Tv = value_linear_project.shape[1]
            query_n_heads = tf.transpose(
                tf.reshape(query_linear_project, [-1, Tq, self.n_head, self.d_k]),
                [0, 2, 1, 3])
            value_n_heads = tf.transpose(
                tf.reshape(value_linear_project, [-1, Tv, self.n_head, self.d_v]),
                [0, 2, 1, 3])
            key_n_heads = tf.transpose(
                tf.reshape(key_linear_project, [-1, Tv, self.n_head, self.d_k]),
                [0, 2, 1, 3])
            mask_extend = None
            if mask:
                mask_extend = [tf.expand_dims(m, -2) \
                                   if m is not None else None \
                                      for m in mask]
            # (batch, n_head, Tq, d_v)
            attention = self.attention(
                [query_n_heads, value_n_heads, key_n_heads],
                mask=mask_extend
            )
            # (batch, Tq, n_head * d_v)
            attention = tf.reshape(tf.transpose(attention, [0, 2, 1, 3]),
                                   [-1, Tq, self.n_head * self.d_v])
        elif multi_head_type == 'for_loop':
            query_n_heads = tf.split(query_linear_project, self.n_head, axis=-1)
            value_n_heads = tf.split(value_linear_project, self.n_head, axis=-1)
            key_n_heads = tf.split(key_linear_project, self.n_head, axis=-1)
            n_head_attentions = [] # n_head *  (batch, Tq, d_v)
            for i in range(self.n_head):
                one_attention = self.attention(
                    [query_n_heads[i], value_n_heads[i], key_n_heads[i]],
                    mask=mask
                )
                n_head_attentions.append(one_attention)
            # (batch, Tq, n_head * d_v)
            attention = tf.concat(n_head_attentions, axis=-1)
        else:
            raise NotImplementedError
        # (batch, Tq, d_model)
        return self.Wo(attention)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, n_head, d_model, dff, dropout=0.0, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model

        # multihead attention, orange block in the model architecture
        self.multi_head_attention = MultiHeadAttention(n_head, d_model)
        # Residual Dropout: We apply dropout [27] to the output of each sub-layer,
        # before it is added to the sub-layer input and normalized.
        self.dropout_attention = tf.keras.layers.Dropout(dropout)

        # Add & Norm, yellow block in the model architecture
        self.add_attention = tf.keras.layers.Add()
        self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed forward, blue block in the model architecture
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout_dense = tf.keras.layers.Dropout(dropout)

        # Add & Norm, yellow block in the model architecture
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None, training=None):
        """
        inputs:
            input embedding, (batch, Te, d_model)

        Return:
            encoder embeding, (batch, Te, d_model)
        """
        assert inputs.shape[-1] == self.d_model, 'last dim of input tensor should be {d_model}'
        # multihead attention, orange block in the model architecture

        attention = self.multi_head_attention(
            [inputs, inputs, inputs], # self attention
            [mask, mask]
        )
        attention = self.dropout_attention(attention, training)
        # Add & Norm, yellow block in the model architecture
        x = self.add_attention([inputs, attention])
        x = self.layer_norm_attention(x)

        # Feed forward, blue block in the model architecture
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training)

        # Add & Norm, yellow block in the model architecture
        x = self.add_dense([x, dense])
        x = self.layer_norm_dense(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, n_head, d_model, dff, dropout=0.0, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        # Masked multihead attention, orange block in the model architecture
        self.masked_multi_head_attention = MultiHeadAttention(n_head, d_model, causal=True)
        # Residual Dropout: We apply dropout [27] to the output of each sub-layer,
        # before it is added to the sub-layer input and normalized.
        self.dropout_masked_attention = tf.keras.layers.Dropout(dropout)

        # Add & Norm, yellow block in the model architecture
        self.add_masked_attention = tf.keras.layers.Add()
        self.layer_norm_masked_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # multihead attention, orange block in the model architecture
        self.multi_head_attention = MultiHeadAttention(n_head, d_model)
        # Residual Dropout: We apply dropout [27] to the output of each sub-layer,
        # before it is added to the sub-layer input and normalized.
        self.dropout_attention = tf.keras.layers.Dropout(dropout)

        # Add & Norm, yellow block in the model architecture
        self.add_attention = tf.keras.layers.Add()
        self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed forward, blue block in the model architecture
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout_dense = tf.keras.layers.Dropout(dropout)

        # Add & Norm, yellow block in the model architecture
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None, training=None):
        """
        inputs:
            (output_embedding, encoder_output)
            output_embedding: (batch, Td, d_model)
            encoder_output:   (batch, Te, d_model_encoder)
        mask:
            (output_mask, encoder_mask)
            output_mask (batch, Td)
            encoder_mask:   (batch, Te)
        """
        output_embedding, encoder_output = inputs
        assert output_embedding.shape[-1] == self.d_model, 'last dim of input tensor should be {d_model}'
        # masked multihead attention, bottom orange block in the decoder

        masked_attention = self.masked_multi_head_attention(  # self attention
            [output_embedding, output_embedding, output_embedding],
            mask=[mask[0], mask[0]] if mask else None # self attention mask
        )
        masked_attention = self.dropout_attention(masked_attention, training)
        # Add & Norm, bottom yellow block in the decoder
        x = self.add_attention([output_embedding, masked_attention])
        x = self.layer_norm_attention(x)

        # multihead attention, middle orange block in the decoder
        # query: the masked_attention output
        # key and value: encoder output
        attention = self.multi_head_attention(
            [x, encoder_output, encoder_output],
            mask=mask # query value mask
        )
        attention = self.dropout_attention(attention, training)
        # Add & Norm, yellow block in the model architecture
        x = self.add_attention([x, attention])
        x = self.layer_norm_attention(x)

        # Feed forward, blue block in the model architecture
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training)
        # Add & Norm, yellow block in the model architecture
        x = self.add_dense([x, dense])
        x = self.layer_norm_dense(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, n_head, d_model, dff,
                 maximum_position_encoding=10000,
                 num_layers=6,
                 dropout=0.0,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, d_model, mask_zero=True)
        self.pos = positional_encoding(maximum_position_encoding, d_model)
        self.encoder_layers = [
            EncoderLayer(n_head=n_head, d_model=d_model, dff=dff, dropout=dropout) \
                for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        """
        inputs:
            (batch, Te)
        """
        x = self.embedding(inputs) # (batch, Te, d_model)
        # In the embedding layers, we multiply those weights by sqrt(d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x[:,i,:] adds a position pos[:,i,:]
        x += self.pos[: , :tf.shape(x)[1], :]
        # In addition, we apply dropout to the sums of the embeddings
        # and the positional encodings in both the encoder and decoder stacks.
        x = self.dropout(x, training=training)

        embedding_mask = self.embedding.compute_mask(inputs)
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x, mask=embedding_mask)
        return x



class Decoder(tf.keras.layers.Layer):
    def __init__(self, target_vocab_size, n_head, d_model, dff,
                 maximum_position_encoding=10000,
                 num_layers=6,
                 dropout=0.0,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size, d_model, mask_zero=True)
        self.pos = positional_encoding(maximum_position_encoding, d_model)
        self.decoder_layers = [
            DecoderLayer(n_head=n_head, d_model=d_model, dff=dff, dropout=dropout) \
                for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        """
        inputs:
            alist [target_input, encoder_output]
        mask:
            encoder mask
        """
        target_input, encoder_output = inputs
        x = self.embedding(target_input) # (batch, Td, d_model)
        # In the embedding layers, we multiply those weights by sqrt(d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x[:,i,:] adds a position pos[:,i,:]
        x += self.pos[: , :tf.shape(x)[1], :]
        # In addition, we apply dropout to the sums of the embeddings
        # and the positional encodings in both the encoder and decoder stacks.
        x = self.dropout(x, training=training)

        embedding_mask = self.embedding.compute_mask(target_input)
        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i]([x, encoder_output],
                                       mask=[embedding_mask, mask])
        return x

# 
# examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
#                                as_supervised=True)
# train_examples, val_examples = examples['train'], examples['validation']
# 
# 
# tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
#     (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
# 
# tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
#     (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
# 
# 
# def encode(lang1, lang2):
#     lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
#         lang1.numpy()) + [tokenizer_pt.vocab_size+1]
# 
#     lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
#       lang2.numpy()) + [tokenizer_en.vocab_size+1]
# 
#     return lang1, lang2
# 
# def tf_encode(pt, en):
#     result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
#     result_pt.set_shape([None])
#     result_en.set_shape([None])
# 
#     return result_pt, result_en
# 
# BUFFER_SIZE = 20000
# BATCH_SIZE = 64
# MAX_LENGTH = 40
# 
# def filter_max_length(x, y, max_length=MAX_LENGTH):
#     return tf.logical_and(tf.size(x) <= max_length,
#                         tf.size(y) <= max_length)
# 
# 
# train_dataset = train_examples.map(tf_encode)
# train_dataset = train_dataset.filter(filter_max_length)
# # cache the dataset to memory to get a speedup while reading from it.
# train_dataset = train_dataset.cache()
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
# train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
# 
# 
# val_dataset = val_examples.map(tf_encode)
# val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)
# 
# 
# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8
# 
# dropout_rate = 0.1
# 
# input_vocab_size = tokenizer_pt.vocab_size + 2
# target_vocab_size = tokenizer_en.vocab_size + 2
# 
# 
# inputs = tf.keras.layers.Input(shape=(None, ))
# targets = tf.keras.layers.Input(shape=(None, ))
# encoder = Encoder(input_vocab_size, num_layers = num_layers, d_model = d_model, n_head = num_heads, dff = dff, dropout = dropout_rate)
# decoder = Decoder(target_vocab_size, num_layers = num_layers, d_model = d_model, n_head = num_heads, dff = dff, dropout = dropout_rate)
# 
# x = encoder(inputs)
# x = decoder([targets, x] , mask = encoder.embedding.compute_mask(inputs))
# #  tf.keras.layers.Masking ??
# x = tf.keras.layers.Dense(target_vocab_size)(x)
# 
# model = tf.keras.models.Model(inputs=[inputs, targets], outputs=x)
# model.summary()
# 
# 
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super(CustomSchedule, self).__init__()
# 
#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)
# 
#         self.warmup_steps = warmup_steps
# 
#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
# 
# optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)
# 
# loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
# 
# def masked_loss(y_true, y_pred):
#     mask = tf.math.logical_not(tf.math.equal(y_true, 0))
#     _loss = loss(y_true, y_pred)
# 
#     mask = tf.cast(mask, dtype=_loss.dtype)
#     _loss *= mask
# 
#     return tf.reduce_sum(_loss)/tf.reduce_sum(mask)
# 
# 
# metrics = [loss, masked_loss, tf.keras.metrics.SparseCategoricalAccuracy()]
# 
# model.compile(optimizer=optimizer, loss = loss, metrics = metrics) # masked_
# 
# def generator(data_set):
#     cnt = 0
#     while cnt < 100:
#         cnt += 1
#         for pt_batch, en_batch in data_set:
#             yield ( [pt_batch , en_batch[:, :-1] ] , en_batch[:, 1:] )
# 
# num_batches = 0
# for (batch, (_,_)) in enumerate(train_dataset):
#     num_batches = batch
# print(num_batches)
# 
# val_batches = 0
# for (batch, (_,_)) in enumerate(val_dataset):
#     val_batches = batch
# print(val_batches)
# 
# history = model.fit(x = generator(train_dataset),
#                     validation_data = generator(val_dataset),
#                     epochs=20, steps_per_epoch = num_batches,
#                     validation_steps = val_batches)