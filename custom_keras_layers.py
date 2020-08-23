import efficientnet.tfkeras as efn
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import collections
import logging
import pdb
logging.basicConfig(level=logging.DEBUG)

class EncodeCordinate(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super(EncodeCordinate, self).__init__()
        _, self.h, self.w, _ = input_shape

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x, y = tf.meshgrid(tf.range(self.w), tf.range(self.h))
        w_loc = tf.one_hot(indices=x, depth=self.w)
        h_loc = tf.one_hot(indices=y, depth=self.h)
        loc = tf.concat([h_loc, w_loc], 2)
        loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
        return tf.concat([inputs, loc], 3)


class SliceRNNInput(tf.keras.layers.Layer):
    def __init__(self):
        super(SliceRNNInput, self).__init__()

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        feature_size = tf.shape(inputs)[-1]
        return tf.reshape(inputs, [batch_size, -1, feature_size])


class ConvBaseLayer(tf.keras.layers.Layer):
    def __init__(self, hparams):
        super(ConvBaseLayer, self).__init__()
        self.hparams = hparams
        if hparams.base_model_name == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'InceptionResNetV2':
            base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB0':
            base_model = efn.EfficientNetB0(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB1':
            base_model = efn.EfficientNetB1(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB2':
            base_model = efn.EfficientNetB2(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB3':
            base_model = efn.EfficientNetB3(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB4':
            base_model = efn.EfficientNetB4(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB5':
            base_model = efn.EfficientNetB5(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB6':
            base_model = efn.EfficientNetB6(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB7':
            base_model = efn.EfficientNetB7(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]

        assert hparams.end_point in base_model_layers, "no {} layer in {}".format(hparams.end_point, hparams.base_model_name)
        conv_tower_output = base_model.get_layer(hparams.end_point).output
        self.conv_model = tf.keras.models.Model(inputs=base_model.input, outputs=conv_tower_output)
        self.conv_out_shape = self.conv_model.predict(np.array([np.zeros(hparams.image_shape)])).shape
        self.encode_cordinate = EncodeCordinate(input_shape=self.conv_out_shape)
        self.slice_rnn_input = SliceRNNInput()

    def call(self, inputs):
        conv_out  = self.conv_model(inputs)
        loc_out   = self.encode_cordinate(conv_out)
        if self.hparams.use_encode_cordinate:
            input_rnn = self.slice_rnn_input(loc_out)
        else:
            input_rnn = self.slice_rnn_input(conv_out)
        return input_rnn


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
                self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.hparams = hparams
        self.dec_units = hparams.rnn_units
        if hparams.rnn_cell == 'gru':
            self.cell = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        elif hparams.rnn_cell == 'lstm':
            self.cell = tf.keras.layers.LSTM(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(hparams.charset_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def initial_hidden_state(self, current_batch_size):
        return tf.zeros((current_batch_size, self.dec_units))

    def call(self, x, hidden, feat_map):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, feat_map)

        x = tf.expand_dims(x, axis=1)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1)

        # passing the concatenated vector to the rnn cell
        if self.hparams.rnn_cell == 'gru':
            output, state = self.cell(x)
        elif self.hparams.rnn_cell == 'lstm':
            output, state, _ = self.cell(x)
 
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights
        


