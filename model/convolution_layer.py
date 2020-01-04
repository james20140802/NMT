"""Implementation of the residual connected convolution layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ResidualConvolutionBlock(tf.keras.layers.Layer):
    """Residual connected convolution block."""
    def __init__(self, n_filters, kernel_size):
        """Initialize the block.

        Args:
          n_filters: Integer, the dimensionality of the output space.
          kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.

        Raise:
          Raise value error if n_filters is not dividable by 4.
        """

        if (n_filters % 4) != 0:
            raise ValueError("Filter number should be divided by 4.")

        super(ResidualConvolutionBlock, self).__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv1 = None
        self.conv2 = None
        self.conv3 = None

        self.relu = None

    def build(self, input_shape):
        """Build the block."""
        self.conv1 = tf.keras.layers.Conv1D(int(self.n_filters/4), 1, padding="same")
        self.conv2 = tf.keras.layers.Conv1D(int(self.n_filters/4), self.kernel_size, padding="same")
        self.conv3 = tf.keras.layers.Conv1D(self.n_filters, 1, padding="same")

        self.relu = tf.keras.layers.ReLU()

    def get_config(self):
        return {
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size
        }

    def call(self, inputs, **kwargs):
        """Return the output of the residual connected convolution block.

        Args:
          inputs: tensor with shape [batch_size, input_length, hidden_size].

        Returns:
          Output of the block.
          float32 tensor with shape [batch_size, length, hidden_size]
        """
        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        x += inputs
        x = self.relu(x)

        return x


class ResidualConvolutionNetwork(tf.keras.Model):
    """Residual connected convolution model."""
    def __init__(self, units, n_filters, kernel_size):
        """Initialize the model.

        Args:
          units: number of convolution layer.
          n_filters: Integer, the dimensionality of the output space.
          kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
        """

        super(ResidualConvolutionNetwork, self).__init__()

        self.units = units
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.convolution_blocks = [ResidualConvolutionBlock(n_filters, kernel_size) for _ in range(units)]

    def get_config(self):
        return {
            "units": self.units,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size
        }

    def call(self, inputs, training=False, mask=None):
        """Return the output of the residual connected convolution model.

        Args:
          inputs: input, tensor with shape [batch_size, input_length, hidden_size].
          training: bool, whether in training mode or not.
          mask: float, tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          Output of the model.
          float32 tensor with shape [batch_size, length, hidden_size]
        """

        x = inputs

        for i in range(self.units):
            x = self.convolution_blocks[i](x)

        return x
