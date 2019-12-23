"""Defines the Transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.embedding_layer import EmbeddingSharedWeights
from model.encoder_layer import Encoder
from model.decoder_layer import Decoder


class Transformer(tf.keras.Model):
    """Transformer model."""
    def __init__(self, params):
        """Initialize layers to build Transformer model.

        Args:
          params: hyper parameter object defining layer sizes, dropout values, etc.
        """

        super(Transformer, self).__init__()

        self.encoder = Encoder(params)

        self.decoder = Decoder(params)

        self.dense = EmbeddingSharedWeights(params["target_vocab_size"], params["hidden_size"])

    def call(self, inputs, training=False, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
        enc_input, dec_input = inputs
        enc_output = self.encoder(inputs, training, enc_padding_mask)

        dec_output = self.decoder((dec_input, enc_output), training, look_ahead_mask, dec_padding_mask)

        output = self.dense(dec_output, mode="linear")

        return output
