"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculate input embeddings and pre-softmax linear"""
    def __init__(self, vocab_size, hidden_size):
        """Specify characteristic parameters of embedding layer

        Args:
            vocab_size: Number of tokens in the embedding
            hidden_size: Dimensionality of the embedding
        """

        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.shared_weights_0 = None
        self.shared_weights_1 = None

    def build(self, input_shape):
        """Build embedding layer"""
        with tf.name_scope("embedding_and_softmax"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.shared_weights_0 = self.add_weight("weights_0",
                                                    shape=[self.vocab_size, self.hidden_size * 2],
                                                    initializer=tf.random_normal_initializer(
                                                      mean=0, stddev=(self.hidden_size * 2) ** 0.5))
            self.shared_weights_1 = self.add_weight("weights_1",
                                                    shape=[self.hidden_size * 2, self.hidden_size],
                                                    initializer=tf.random_normal_initializer(
                                                      mean=0, stddev=self.hidden_size ** 0.5))

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
        }

    def call(self, inputs, mode="embedding"):
        """Get token embeddings of inputs

        Args:
            inputs: An int64 tensor with [batch_size, length]
            mode: string, a valid value is one of "embedding" and "linear"
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
            shape [batch_size, length, embedding_size]; (2) mode == "linear", output
            linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raise:
            ValueError: if mode is not valid.
        """

        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError('mode {} is not valid. it should be either "embedding" or "linear"'.format(mode))

    def _embedding(self, inputs):
        """Applies embedding based on inputs tensor.

        Args:
            inputs: A int32 tensor with shape [batch_size, length]
        Returns:
            float32 tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("embedding"):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]
            # Create binary mask of size [batch_size, length]
            x = tf.keras.utils.to_categorical(inputs, num_classes=self.vocab_size)
            x = tf.matmul(x, self.shared_weights_0)
            embeddings = tf.matmul(x, self.shared_weights_1)
            mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
            embeddings *= tf.expand_dims(mask, -1)
            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5

            return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.

        Args:
          inputs: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
          float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]

            x = tf.reshape(inputs, [-1, self.hidden_size])
            x = tf.matmul(x, self.shared_weights_1, transpose_b=True)
            logits = tf.matmul(x, self.shared_weights_0, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])
