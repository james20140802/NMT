"""Implementation of feed forward network with convolution"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FeedForwardNetwork(tf.keras.layers.Layer):
    """Feed forward network with convolution"""
    def __init__(self, hidden_size, filter_size, relu_dropout):
        """Initialize FeedForwardNetwork.

        Args:
          hidden_size: int, output dim of convolution layer.
          filter_size: int, filter size for the inner (first) convolution layer.
          relu_dropout: float, dropout rate for training.
        """

        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout

        self.convolution_1 = None
        self.convolution_2 = None

    def build(self, input_shape):
        """Build the layer"""
        self.convolution_1 = tf.keras.layers.Conv1D(self.filter_size, 1, use_bias=True)
        self.convolution_2 = tf.keras.layers.Conv1D(self.hidden_size, 1, use_bias=True)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout
        }

    def call(self, inputs, training=False):
        """Return outputs of feed forward network

        Args:
             inputs: tensor with shape [batch_size, length, hidden_size]
             training: bool, whether in training mode or not
        Returns:
            Output of the feed forward network
            tensor with shape [batch_size, length, hidden_size]
        """

        output = self.convolution_1(inputs)
        if training:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = self.convolution_2(output)

        return output
