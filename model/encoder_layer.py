"""Implementation of encoder layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.embedding_layer import EmbeddingSharedWeights
from model.attention_layer import Attention
from model.ffn_layer import FeedForwardNetwork


class EncoderLayer(tf.keras.layers.Layer):
    """Encoder layer."""
    def __init__(self, params):
        """Initialize encoder layer.

        Args:
          params: hyper parameter object defining layer sizes, dropout values, etc.
        """

        super(EncoderLayer, self).__init__()
        self.hidden_size = params["hidden_size"]
        self.num_heads = params["num_heads"]
        self.kernel_size = params["kernel_size"]
        self.filter_size = params["filter_size"]
        self.dropout_rate = params["dropout_rate"]

        self.attention = None
        self.ffn = None
        self.layer_norm1 = None
        self.layer_norm2 = None

    def build(self, input_shape):
        """Build the layer."""
        self.attention = Attention(self.hidden_size, self.num_heads, self.kernel_size, self.dropout_rate)
        self.ffn = FeedForwardNetwork(self.hidden_size, self.filter_size, self.dropout_rate)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=True, mask=None):
        """Return the output of the encoder layer.

        Args:
          inputs: tensor with shape [batch_size, input_length, hidden_size].
          training: bool, whether in training mode or not.
          mask: float, tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          Output of encoder layer
          float32 tensor with shape [batch_size, input_length, hidden_size].
        """

        x = self.attention((inputs, inputs), training=training, mask=mask)
        x = self.layer_norm1(x + inputs)

        y = self.ffn(x, training=training)
        y = self.layer_norm2(x + y)

        return y


class Encoder(tf.keras.Model):
    """Encoder model."""
    def __init__(self, params):
        """Initialize the encoder model.

        Args:
          params: hyper parameter object defining layer sizes, dropout values, etc.
        """

        super(Encoder, self).__init__()
        self.num_layers = params["num_layers"]
        self.hidden_size = params["hidden_size"]
        self.num_heads = params["num_heads"]
        self.kernel_size = params["kernel_size"]
        self.filter_size = params["filter_size"]
        self.vocab_size = params["input_vocab_size"]
        self.maximum_position_encoding = params["input_maximum_position_encoding"]
        self.dropout_rate = params["dropout_rate"]

        self.embedding = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)
        self.pos_encoding = EmbeddingSharedWeights.positional_encoding(self.maximum_position_encoding, self.hidden_size)
        self.encoder_layers = [EncoderLayer(params) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False, mask=None):
        """Return the output of the encoder.

        Args:
          inputs: input, tensor with shape [batch_size, input_length].
          training: bool, whether in training mode or not.
          mask: float, tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k). Defaults to None.

        Return:
          Output of the encoder.
          float32 tensor with shape [batch_size, length, hidden_size]

        Raises:
          NotImplementedError: If try to use padded decode method on CPU/GPUs.
        """

        seq_len = tf.shape(inputs)[1]

        emb = self.embedding(inputs, mode="embedding")
        emb *= tf.math.sqrt(tf.cast(self.hidden_size, tf.float32))
        emb += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(emb, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, mask)

        return x

