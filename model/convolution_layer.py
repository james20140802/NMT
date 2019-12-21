"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ConvolutionBlock(tf.keras.layers.Layer):
    """Calculate convolution over sentence"""
    def __init__(self, hidden_size, kernel_size, stride=1, is_first_layer=False):
        """Specify characteristic parameters of embedding layer

        Args:
            hidden_size: Integer, the dimensionality of the output space
            kernel_size: An integer or tuple/list of a single integer, specifying the length of the convolution window

        Raise:
            If hidden size is not dividable by 4
        """

        super(ConvolutionBlock, self).__init__()

        if hidden_size % 4 != 0:
            raise Exception("Hidden size should be dividable by 4")

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_first_layer = is_first_layer

        self.res_branch1 = None

        self.bn_branch1 = None

        self.res_branch2a = None
        self.res_branch2b = None
        self.res_branch2c = None

        self.bn_branch_2a = None
        self.bn_branch_2b = None
        self.bn_branch_2c = None

        self.relu = None

    def build(self, input_shape):
        """Build convolution layer"""
        self.res_branch2a = tf.keras.layers.Conv1D(int(self.hidden_size / 4), 1,  strides=self.stride, padding='same')
        self.res_branch2b = tf.keras.layers.Conv1D(int(self.hidden_size / 4), self.kernel_size, padding='same')
        self.res_branch2c = tf.keras.layers.Conv1D(self.hidden_size, 1, padding="same")

        self.bn_branch_2a = tf.keras.layers.BatchNormalization()
        self.bn_branch_2b = tf.keras.layers.BatchNormalization()
        self.bn_branch_2c = tf.keras.layers.BatchNormalization()

        self.res_branch1 = tf.keras.layers.Conv1D(self.hidden_size, 1, strides=self.stride, padding="same")
        self.bn_branch1 = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "is_first_layer": self.is_first_layer
        }

    def call(self, inputs, **kwargs):
        """Get latent vector by convolution network with residual connection

        Args:
            inputs: A float32 tensor with [batch_size, length, hidden_size]
        Returns:
            outputs: latent vector, float vector with shape [batch_size, length, hidden_size]
        """

        residual = self.res_branch2a(inputs)
        residual = self.bn_branch_2a(residual)
        residual = self.relu(residual)

        residual = self.res_branch2b(residual)
        residual = self.bn_branch_2b(residual)
        residual = self.relu(residual)

        residual = self.res_branch2c(residual)
        residual = self.bn_branch_2c(residual)

        if self.stride != 1 or self.is_first_layer:
            identity_mapping = self.res_branch1(inputs)
            identity_mapping = self.bn_branch1(identity_mapping)
        else:
            identity_mapping = inputs

        outputs = residual + identity_mapping
        outputs = self.relu(outputs)

        return outputs
