"""Defines SubTokenizer class to encode and decode strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow_datasets as tfds
import tensorflow as tf

PAD = "<pad>"
PAD_ID = 0
EOS = "<EOS>"
EOS_ID = 1
RESERVED_TOKENS = [PAD, EOS]


class SubTokenizer(object):
    """Encodes and decodes strings to/from integer IDs."""

    def __init__(self, filename, reserved_tokens=None):
        """Initializes class, creating a vocab file if data_files is provided."""
        logging.info("Initializing SubTokenizer from file %s." % filename)

        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS

        self.tokenizer = _load_tokenizer(filename)
        self.reserved_tokens = reserved_tokens

    @staticmethod
    def init_from_files(tokenizer_filename, files, target_vocab_size, max_subword_length=20,
                        file_byte_limit=1e6, reserved_tokens=None):
        """Create subtoken vocabulary based on files, and save tokenizer to file.

        Args:
          tokenizer_filename:  String name of  file to store tokenizer.
          files: List of file paths that will be used to generate vocabulary.
          target_vocab_size: target vocabulary size to generate.
          max_subword_length: int, maximum length of a subword.
          file_byte_limit: (Default 1e6) Maximum number of bytes of sample text that
                                         will be drawn from the files.
          reserved_tokens: List of string tokens that are guaranteed to be at
                           the beginning of the subtoken vocabulary list.

        Returns:
          SubTokenizer object.
        """
        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS

        if tf.io.gfile.exists(_filename(tokenizer_filename)):
            logging.info("Vocab file already exists (%s)" % tokenizer_filename)
        else:
            logging.info("Begin steps to create subtoken vocabulary...")
            corpus_generator = _generator(files, file_byte_limit)
            tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus_generator, target_vocab_size,
                                                                                max_subword_length,
                                                                                reserved_tokens=reserved_tokens)
            logging.info("Generated vocabulary with %d subtokens." % tokenizer.vocab_size)
            _save_tokenizer(tokenizer_filename, tokenizer)

        return SubTokenizer(tokenizer_filename)

    def encode(self, raw_string, add_eos=False):
        """Encodes a string into a list of int subtoken ids."""
        ret = self.tokenizer.encode(raw_string)

        if add_eos:
            ret.append(EOS_ID)

        return ret

    def decode(self, subtokens):
        """Converts list of int subtokens ids into a string."""
        ret = self.tokenizer.decode(subtokens)

        return ret


def _load_tokenizer(filename_prefix):
    """Load tokenizer."""
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix)

    return tokenizer


def _save_tokenizer(filename_prefix, tokenizer):
    tokenizer.save_to_file(filename_prefix)


def _generator(files, file_byte_limit=1e6):
    """Return generator that yield string from files.

    Args:
      files: List of file paths.
      file_byte_limit: Max number of bytes that will be read from each file.

    Returns:
      Generator yielding string.
    """

    for file_path in files:
        with tf.io.gfile.GFile(file_path, 'r') as reader:
            file_byte_budget = file_byte_limit
            counter = 0
            lines_to_skip = int(reader.size() / (file_byte_budget * 2))

            for line in reader:
                if counter < lines_to_skip:
                    counter += 1
                else:
                    if file_byte_budget < 0:
                        break
                    line = line.strip()

                    yield line


def _filename(filename_prefix):
    return filename_prefix + ".subwords"
