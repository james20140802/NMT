from model.convolution_layer import ResNetBlock

import tensorflow as tf

HIDDEN_SIZE = 512
STRIDE = 2
KERNEL_SIZE = 3
LENGTH = 32
BATCH_SIZE = 64

inputs = tf.ones([BATCH_SIZE, LENGTH, HIDDEN_SIZE])

block_1 = ResNetBlock(HIDDEN_SIZE, KERNEL_SIZE, 1)
block_2 = ResNetBlock(HIDDEN_SIZE * 2, KERNEL_SIZE, 2)
block_3 = ResNetBlock(HIDDEN_SIZE * 4, KERNEL_SIZE, 2)

x = block_1(inputs)
x = block_2(x)
x = block_3(x)

print(tf.shape(x))
