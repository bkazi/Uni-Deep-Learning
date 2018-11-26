from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from utils import get_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max-steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer(
    'batch-size', 16, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 5e-5,
                          'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'input-width', 80, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'input-height', 80, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'num-classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))


def main():
    train_data, train_labels = get_data()

    with tf.variable_scope('inputs'):
        x = tf.placeholder(
            tf.float32, [None, FLAGS.img_width, FLAGS.img_height])
        y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])


if __name__ == '__main__':
    tf.app.run(main=main)
