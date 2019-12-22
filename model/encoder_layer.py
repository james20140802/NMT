"""Implementation of encoder layer"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.embedding_layer import EmbeddingSharedWeights
from model.attention_layer import Attention
from model.ffn_layer import FeedForwardNetwork


class EncoderLayer(tf.keras.layers.Layer):
    """Encoder layer"""
    def __init__(self, hidden_size, num_heads, kernel_size, filter_size, dropout_rate):
        """Initialize encoder layer

        Args:
            hidden_size: int, output dim of hidden layer.
            num_heads: int, number of heads to repeat the same attention structure.
            kernel_size: int, the length of the 1D convolution window.
            filter_size: int, filter size of feed forward network
            dropout_rate: float, dropout rate for training
        """

        super(EncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate

        self.attention = None
        self.ffn = None
        self.layer_norm1 = None
        self.layer_norm2 = None

    def build(self, input_shape):
        """Build the layer"""
        self.attention = Attention(self.hidden_size, self.num_heads, self.kernel_size, self.dropout_rate)
        self.ffn = FeedForwardNetwork(self.hidden_size, self.filter_size, self.dropout_rate)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=True, mask=None):
        """Return the output of the encoder layer.

        Args:
          inputs: tensor with shape [batch_size, input_length, hidden_size]
          training: boolean, whether in training mode or not.
          mask: Float tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          Output of encoder layer.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """

        x = self.attention((inputs, inputs), training=training, mask=mask)
        x = self.layer_norm1(x + inputs)

        y = self.ffn(x, training=training)
        y = self.layer_norm2(x + y)

        return y


class Encoder(tf.keras.Model):
    """Encoder model"""
    def __init__(self, num_layers, hidden_size, num_heads, kernel_size,
                 filter_size, vocab_size, maximum_position_encoding, dropout_rate=.2):
        """Initialize encoder

        Args:
            num_layers: int, number of encoder layer
            hidden_size: int, output dim of hidden layer.
            num_heads: int, number of heads to repeat the same attention structure.
            kernel_size: int, the length of the 1D convolution window.
            filter_size: int, filter size of feed forward network.
            vocab_size: Number of tokens in the embedding
            maximum_position_encoding: int for positional encoding
            dropout_rate: float, dropout rate for training. Default value is 0.2
        """

        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.vocab_size = vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate = dropout_rate

        self.embedding = None
        self.pos_encoding = None
        self.encoder_layers = None

    def build(self, input_shape):
        """Build the model"""
        self.embedding = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)
        self.pos_encoding = EmbeddingSharedWeights.positional_encoding(self.maximum_position_encoding, self.hidden_size)
        self.encoder_layers = [EncoderLayer(self.hidden_size, self.num_heads, self.kernel_size,
                                            self.filter_size, self.dropout_rate)
                               for _ in range(self.num_layers)]

    def call(self, inputs, training=False, mask=None):
        """Return the output of the encoder.

        Args:
            inputs: input tensor with shape [batch_size, input_length].
            training: bool, whether in training mode or not.
            mask: Float tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k). Defaults to None.

        Return:
            Output of the encoder.
            float32 tensor with shape [batch_size, length, hidden_sie]

         Raises:
            NotImplementedError: If try to use padded decode method on CPU/GPUs.
        """

        seq_len = tf.shape(inputs)[1]

        emb = self.embedding(inputs, mode="embedding")
        emb *= tf.math.sqrt(tf.cast(self.hidden_size, tf.float32))
        emb += self.pos_encoding[:, :seq_len, :]

        if training:
            x = tf.nn.dropout(emb, rate=self.dropout_rate)
        else:
            x = emb

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, mask)

        return x

