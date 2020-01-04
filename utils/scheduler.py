"""Defines Scheduler class which is custom learning rate scheduler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Scheduler for training."""

    def __init__(self, hidden_size, warmup_steps=4000):
        """Initializes class."""
        super(Scheduler, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_size = tf.cast(self.hidden_size, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "warmup_steps": self.warmup_steps
        }
