"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ConvolutionBlock(tf.keras.layers.Layer):
    """Calculate convolution over sentence"""
    def __init__(self, hidden_size, kernel_size, dropout_p=.5):
        """Specify characteristic parameters of embedding layer

        Args:
            hidden_size: Integer, the dimensionality of the output space
            kernel_size: An integer specifying the width of the convolution window
        """

        super(ConvolutionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.convolution_a1 = None
        self.convolution_a2 = None
        self.convolution_b1 = None
        self.convolution_b2 = None

        self.batch_normalization_1 = None
        self.batch_normalization_2 = None

        self.relu = None

    def build(self, input_shape):
        """Build convolution layer"""
        self.convolution_a1 = tf.keras.layers.Conv2D(self.hidden_size, (self.kernel_size, self.hidden_size),
                                                     padding='same')
        self.convolution_a2 = tf.keras.layers.Conv2D(self.hidden_size, (self.kernel_size, self.hidden_size),
                                                     padding='same')
        self.convolution_b1 = tf.keras.layers.Conv2D(self.hidden_size, 1)
        self.convolution_b2 = tf.keras.layers.Conv2D(self.hidden_size, 1)

        self.batch_normalization_1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization_2 = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "kernel_size": self.kernel_size
        }

    def call(self, inputs, **kwargs):
        """Get latent vector by convolution network with residual connection

        Args:
            inputs: A float32 tensor with [batch_size, length, hidden_size]
        Returns:
            outputs: latent vector, float vector with shape [batch_size, length, hidden_size]
        """

        x = self.convolution_a1(tf.expand_dims(inputs, 1))
        x = self.relu(x)
        x = self.convolution_a2(x)

        residual = self.batch_normalization_1(inputs)
        residual = self.relu(residual)
        residual = self.convolution_b1(residual)
        residual = self.batch_normalization_2(residual)
        residual = self.relu(residual)
        residual = self.convolution_b2(residual)

        x += residual

        return x
