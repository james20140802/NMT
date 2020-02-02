"""Implementation of multi-headed attention with convolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, attention_dropout):
        """Initialize Attention.

       Args:
         hidden_size: int, output dim of hidden layer.
         attention_dropout: float, dropout rate inside attention for training.
       """

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dropout = attention_dropout

        self.dropout = None

    def build(self, input_shape):
        self.dropout = tf.keras.layers.Dropout(self.attention_dropout)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "attention_dropout": self.attention_dropout
        }

    def call(self, query_input, source_input, training=False, mask=None):
        """Apply attention mechanism to query_input and source_input.

        Args:
          query_input: A tensor with shape [batch_size, length_query, hidden_size].
          source_input: A tensor with shape [batch_size, length_source, hidden_size].
          training: bool, whether in training mode or not.
          mask: float, tensor with shape that can be broadcast to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          Attention layer output with shape [batch_size, length_query, hidden_size].
        """
        logits = tf.matmul(query_input, source_input, transpose_b=True)

        dk = tf.cast(tf.shape(query_input)[-1], tf.float32)
        scaled_attention_logits = logits / tf.math.sqrt(dk)

        if mask:
            scaled_attention_logits += (mask * -1e9)

        weights = tf.nn.softmax(scaled_attention_logits)
        weights = self.dropout(weights, training=training)

        output = tf.matmul(weights, source_input)

        return output
