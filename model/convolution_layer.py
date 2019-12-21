"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ResNetIdentityBlock(tf.keras.Model):
    """Calculate convolution over sentence"""
    def __init__(self, hidden_size, kernel_size, stride=1, is_first_layer=False):
        """Specify characteristic parameters of convolution block and build the block

        Args:
            hidden_size: Integer, the dimensionality of the output space
            kernel_size: An integer or tuple/list of a single integer, specifying the length of the convolution window
            stride: An integer or tuple/list of a single integer, specifying the stride length of the convolution
            is_first_layer: Bool, if it is true, the block uses 1X1 convolution in identity mapping

        Raise:
            If hidden size is not dividable by 4
        """

        super(ResNetIdentityBlock, self).__init__()

        if hidden_size % 4 != 0:
            raise Exception("Hidden size should be dividable by 4")

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_first_layer = is_first_layer

        self.res_branch2a = tf.keras.layers.Conv1D(int(self.hidden_size / 4), 1, strides=self.stride, padding='same')
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


class ResNetBlock(tf.keras.Model):
    """Construct ResNet block using ResNetIdentityBlock"""
    def __init__(self, hidden_size, kernel_size, stride=1):
        """Specify characteristic parameters of ResNet block and build the block

        Args:
            hidden_size: Integer, the dimensionality of the output space
            kernel_size: An integer or tuple/list of a single integer, specifying the length of the convolution window
            stride: An integer or tuple/list of a single integer, specifying the stride length of the convolution

        """
        super(ResNetBlock, self).__init__()

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride

        self.block_1 = ResNetIdentityBlock(hidden_size, kernel_size, stride=stride, is_first_layer=True)
        self.block_2 = ResNetIdentityBlock(hidden_size, kernel_size)
        self.block_3 = ResNetIdentityBlock(hidden_size, kernel_size)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "kernel_size": self.kernel_size,
            "stride": self.stride
        }

    def call(self, inputs, **kwargs):
        """Get latent vector by convolution network with residual connection

        Args:
            inputs: A float32 tensor with [batch_size, length, hidden_size]
        Returns:
            outputs: latent vector, float vector with shape [batch_size, length, hidden_size]
        """

        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)

        return x
