"""Defines of the voc2voc model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.embedding_layer import EmbeddingSharedWeights
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
        self.dropout_rate = params["dropout_rate"]

        self.input_embedding = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)
        self.target_embedding = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)

        self.attention = Attention(self.hidden_size, self.dropout_rate)

        self.language_model = LanguageModel(self.units, self.hidden_size, self.kernel_size, self.dropout_rate)

    def call(self, inputs, training=False, look_ahead_mask=None, input_padding_mask=None, target_padding_mask=None):
        inputs, targets = inputs

        input_embedding = self.input_embedding(inputs)
        target_embedding = self.target_embedding(targets)

        if input_padding_mask:
            input_embedding *= (1 - input_padding_mask)

        if target_padding_mask:
            target_embedding *= (1 - target_padding_mask)

        attention_output = self.attention((target_embedding, input_embedding), training=training)

        output = self.language_model(attention_output)

        output = self.target_embedding(output, "linear")

        output = tf.nn.softmax(output)

        return output
