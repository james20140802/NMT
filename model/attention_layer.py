"""Implementation of multi-headed attention with convolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, kernel_size, attention_dropout):
        """Initialize Attention.

       Args:
         hidden_size: int, output dim of hidden layer.
         num_heads: int, number of heads to repeat the same attention structure.
         kernel_size: int, the length of the 1D convolution window.
         attention_dropout: float, dropout rate inside attention for training.
       """
        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({}).".format(hidden_size, num_heads))

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.attention_dropout = attention_dropout

        self.depth = hidden_size // self.num_heads

        self.query_convolution = None
        self.key_convolution = None
        self.value_convolution = None

        self.output_dense = None

    def build(self, input_shape):
        """Builds the layer."""

        self.query_convolution = tf.keras.layers.Conv1D(self.hidden_size, self.kernel_size, padding="same")
        self.key_convolution = tf.keras.layers.Conv1D(self.hidden_size, self.kernel_size, padding="same")
        self.value_convolution = tf.keras.layers.Conv1D(self.hidden_size, self.kernel_size, padding="same")

        self.output_dense = tf.keras.layers.Dense(self.hidden_size)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "kernel_size": self.kernel_size,
            "attention_dropout": self.attention_dropout
        }

    def split_heads(self, inputs):
        # Split the head
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=True, mask=None):
        """Apply attention mechanism to query_input and source_input.

        Args:
          inputs: A tuple Contain two tensor
              query_input: A tensor with shape [batch_size, length_query, hidden_size].
              source_input: A tensor with shape [batch_size, length_source, hidden_size].
          training: A bool, whether in training mode or not.
          mask: Float tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          Attention layer output with shape [batch_size, length_query, hidden_size]
        """
        query_input, source_input = inputs
        batch_size = tf.shape(query_input)[0]

        query = self.query_convolution(query_input)
        key = self.key_convolution(source_input)
        value = self.value_convolution(source_input)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Scale query to prevent the dot product between query and key from growing
        # too large.
        depth = (self.hidden_size // self.num_heads)
        query *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(query, key, transpose_b=True)

        if mask is not None:
            logits += (mask * -1e9)

        weights = tf.nn.softmax(logits, name="attention_weights")

        if training:
            weights = tf.nn.dropout(weights, rate=self.attention_dropout)

        attention_output = tf.matmul(weights, value)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.hidden_size))

        attention_output = self.output_dense(attention_output)
        return attention_output
