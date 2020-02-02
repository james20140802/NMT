"""Defines of the voc2voc model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.embedding_layer import EmbeddingSharedWeights
from model.embedding_layer import ELMo
from model.language_model import LanguageModel
from model.attention_layer import Attention


class Voc2Voc(tf.keras.Model):
    """Voc2Voc model."""

    def __init__(self, params):
        """Initialize layers to build Transformer model.

        Args:
          params: hyper parameter object defining layer sizes, dropout values, etc.
        """

        super(Voc2Voc, self).__init__()

        self.vocab_size = params["vocab_size"]
        self.hidden_size = params["hidden_size"]
        self.kernel_size = params["kernel_size"]
        self.units = params["units"]
        self.elmo_units = params["elmo_units"]
        self.dropout_rate = params["dropout_rate"]

        self.input_embedding = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)
        self.target_embedding = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)

        self.input_elmo = ELMo(self.elmo_units, self.hidden_size)
        self.target_elmo = ELMo(self.elmo_units, self.hidden_size)

        self.attention = Attention(self.hidden_size, self.dropout_rate)

        self.language_model = LanguageModel(self.units, self.hidden_size, self.kernel_size, self.dropout_rate)

    def call(self, inputs, targets, training=False, look_ahead_mask=None,
             input_padding_mask=None, target_padding_mask=None):

        batch_size = tf.shape(inputs)[0]

        inputs_length = tf.shape(inputs)[1]
        targets_length = tf.shape(targets)[1]

        input_embedding = self.input_embedding(inputs)
        target_embedding = self.target_embedding(targets)

        input_embedding = self.input_elmo(input_embedding)
        target_embedding = self.target_elmo(target_embedding)

        if input_padding_mask is not None:
            input_embedding += tf.broadcast_to(tf.reshape(input_padding_mask, [batch_size, inputs_length, 1]) * 1e-9,
                                               [batch_size, inputs_length, self.hidden_size])

        if target_padding_mask is not None:
            target_embedding += tf.broadcast_to(tf.reshape(target_padding_mask, [batch_size, targets_length, 1]) * 1e-9,
                                                [batch_size, targets_length, self.hidden_size])

        attention_output = self.attention(target_embedding, input_embedding, training=training)

        output = self.target_embedding(attention_output, "linear")

        output = tf.nn.softmax(output)

        return output
