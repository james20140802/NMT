"""Implementation of the recurrenct layer with residual connection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ResidualLSTMBlock(tf.keras.layers.Layer):
    """LSTM block with residual connection."""
    def __init__(self, hidden_size):
        """Initialize the block.

        Args:
          hidden_size: int, hidden size of the LSTM
        """

        super(ResidualLSTMBlock, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = None

    def build(self, input_shape):
        """Build the block."""
        self.lstm = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size
        }

    def call(self, inputs, **kwargs):
        """Return the output of the residual connected convolution block.

        Args:
          inputs: tensor with shape [batch_size, input_length, hidden_size].

        Returns:
          Output of the block.
          float32 tensor with shape [batch_size, length, hidden_size]
        """

        x = self.lstm(inputs)

        x += inputs
        x = tf.math.tanh(x)

        return x


class ResidualLSTMNetwork(tf.keras.Model):
    """Residual connected LSTM model."""

    def __init__(self, units, hidden_size):
        """Initialize the model.

        Args:
          units: number of convolution layer.
          hidden_size: Integer, the dimensionality of the output space.
        """

        super(ResidualLSTMNetwork, self).__init__()

        self.units = units
        self.hidden_size = hidden_size

        self.lstm_blocks = [ResidualLSTMBlock(hidden_size) for _ in range(units)]

    def get_config(self):
        return {
            "units": self.units,
            "hidden_size": self.hidden_size
        }

    def call(self, inputs, training=False, mask=None):
        """Return the output of the residual connected LSTM model.

        Args:
          inputs: input, tensor with shape [batch_size, input_length, hidden_size].
          training: bool, whether in training mode or not.
          mask: float, tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          Output of the model.s
          float32 tensor with shape [batch_size, length, hidden_size]
        """

        for i in range(self.units):
            inputs = self.lstm_blocks[i](inputs)

        return inputs
