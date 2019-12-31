"""Implementation of language model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.attention_layer import Attention


class LanguageModel(tf.keras.Model):
    """Language model for correcting awkward expression."""
    def __init__(self):
        """Initialize layers to build the language model.

        Args:

        """
        super(LanguageModel, self).__init__()
