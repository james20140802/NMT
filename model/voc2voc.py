"""Defines of the voc2voc model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.embedding_layer import EmbeddingSharedWeights


class Voc2Voc(tf.keras.Model):
    """Voc2Voc model."""

    def __init__(self, params):
        """Initialize layers to build Transformer model.

        Args:
          params: hyper parameter object defining layer sizes, dropout values, etc.
        """

        super(Voc2Voc, self).__init__()
