"""Defines of the voc2voc model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.embedding_layer import EmbeddingSharedWeights
from model.language_model import LanguageModel


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

        self.input_embedding = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)
        self.target_embedding = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)

        self.language_model = LanguageModel(self.units, self.hidden_size, self.kernel_size)

    def call(self, inputs, training=None, look_ahead_mask=None, input_padding_mask=None, target_padding_mask=None):
        inputs, targets = inputs

        input_embedding = self.input_embedding(inputs)
        target_embedding = self.target_embedding(targets)

        if input_padding_mask:
            input_embedding *= (1 - input_padding_mask)

        if target_padding_mask:
            target_embedding *= (1 - target_padding_mask)

        logits = tf.matmul(target_embedding, input_embedding, transpose_b=True)

        dk = tf.cast(tf.shape(input_embedding)[-1], tf.float32)
        scaled_attention_logits = logits / tf.math.sqrt(dk)

        if look_ahead_mask:
            scaled_attention_logits += (look_ahead_mask * -1e9)

        weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        attention_output = tf.matmul(weights, input_embedding)

        output = self.language_model(attention_output)

        return output
