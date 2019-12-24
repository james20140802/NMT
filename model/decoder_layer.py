""""Implementation of decoder layer and decoder model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.attention_layer import Attention
from model.ffn_layer import FeedForwardNetwork
from model.embedding_layer import EmbeddingSharedWeights


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder layer."""
    def __init__(self, params):
        """Initialize decoder layer.

       Args:
          params: hyper parameter object defining layer sizes, dropout values, etc.
       """

        super(DecoderLayer, self).__init__()
        self.hidden_size = params["hidden_size"]
        self.num_heads = params["num_heads"]
        self.kernel_size = params["kernel_size"]
        self.filter_size = params["filter_size"]
        self.dropout_rate = params["dropout_rate"]

        self.attention1 = None
        self.attention2 = None

        self.ffn = None

        self.layer_norm1 = None
        self.layer_norm2 = None
        self.layer_norm3 = None

    def build(self, input_shape):
        """Build the layer"""
        self.attention1 = Attention(self.hidden_size, self.num_heads, self.kernel_size, self.dropout_rate)
        self.attention2 = Attention(self.hidden_size, self.num_heads, self.kernel_size, self.dropout_rate)

        self.ffn = FeedForwardNetwork(self.hidden_size, self.filter_size, self.dropout_rate)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False, look_ahead_mask=None, padding_mask=None):
        """Return the output of the decoder layer.

        Args:
          inputs: tensor with shape [batch_size, input_length, hidden_size]
          training: bool, whether in training mode or not.
          look_ahead_mask: float, tensor for preventing decoder to look ahead of sequence during training
                           Default to None
          padding_mask: float, tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k).
                        Defaults to None.

        Returns:
          Output of decoder layer.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        x, enc_output = inputs

        attn1 = self.attention1((x, x), training=training, mask=look_ahead_mask)
        out1 = self.layer_norm1(x + attn1)

        attn2 = self.attention2((out1, enc_output), training=training, mask=padding_mask)
        out2 = self.layer_norm2(out1 + attn2)

        ffn = self.ffn(out2, training=training)
        out3 = self.layer_norm3(ffn + out2)

        return out3


class Decoder(tf.keras.Model):
    """Decoder model."""

    def __init__(self, params):
        """Initialize decoder.

        Args:
          params: hyper parameter object defining layer sizes, dropout values, etc.
        """

        super(Decoder, self).__init__()
        self.num_layers = params["num_layers"]
        self.hidden_size = params["hidden_size"]
        self.num_heads = params["num_heads"]
        self.kernel_size = params["kernel_size"]
        self.filter_size = params["filter_size"]
        self.vocab_size = params["target_vocab_size"]
        self.maximum_position_encoding = params["target_maximum_position_encoding"]
        self.dropout_rate = params["dropout_rate"]

        self.embedding = EmbeddingSharedWeights(self.vocab_size, self.hidden_size)
        self.pos_encoding = EmbeddingSharedWeights.positional_encoding(self.maximum_position_encoding, self.hidden_size)
        self.decoder_layers = [DecoderLayer(params) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False, look_ahead_mask=None, padding_mask=None):
        """Return the output of the decoder.

        Args:
          inputs: input tensor with shape [batch_size, input_length] and encoder output.
          training: bool, whether in training mode or not.
          look_ahead_mask: float, tensor for preventing decoder to look ahead of sequence during training.
                            Default value is None.
          padding_mask: float, tensor with shape that can be broadcast  to (..., seq_len_q, seq_len_k).
                        Defaults to None.

        Return:
          Output of the decoder
          float32 tensor with shape [batch_size, length, hidden_size].
        """

        x, enc_output = inputs

        seq_len = tf.shape(x)[1]

        emb = self.embedding(x, mode="embedding")
        emb *= tf.math.sqrt(tf.cast(self.hidden_size, tf.float32))
        emb += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(emb, training=training)

        for i in range(self.num_layers):
            x = self.decoder_layers[i]((x, enc_output), training, look_ahead_mask, padding_mask)

        return x


