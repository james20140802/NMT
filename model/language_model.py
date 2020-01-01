"""Implementation of language model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.convolution_layer import ResidualConvolutionNetwork
from model.recurrent_layer import ResidualLSTMNetwork


class LanguageModel(tf.keras.Model):
    """Language model for correcting awkward expression."""
    def __init__(self, units, hidden_size, kernel_size):
        """Initialize layers to build the language model.

        Args:
          units: number of convolution layer.
          hidden_size: Integer, the dimensionality of the output space.
          kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
        """
        super(LanguageModel, self).__init__()

        self.units = units
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.convolution = ResidualConvolutionNetwork(units, hidden_size, kernel_size)
        self.lstm = ResidualLSTMNetwork(units, hidden_size)

    def get_config(self):
        return {
            "units": self.units,
            "hidden_size": self.hidden_size,
            "kernel_size": self.kernel_size
        }

    def call(self, inputs, training=None, mask=None):
        """Return the output of the language model.

        Args:
          inputs:  input, tensor with shape [batch_size, input_length, hidden_size].
          training: bool, whether in training mode or not.
          mask: float, tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          Output of the language model.
          float32 tensor with shape [batch_size, length, hidden_size]
        """

        x = self.convolution(inputs)
        y = self.lstm(inputs)

        output = tf.math.softmax(x + y)

        return output
