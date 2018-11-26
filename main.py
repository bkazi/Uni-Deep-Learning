from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from utils import get_data
from shallow_nn import shallow_nn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max-steps', 100,
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
    'input-width', 80, 'Input width (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'input-height', 80, 'Input height (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'input-channels', 1, 'Input channels (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'num-classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))


def main(_):

    with tf.variable_scope('inputs'):
        x = tf.placeholder(
            tf.float32, [None, FLAGS.input_width, FLAGS.input_height, FLAGS.input_channels])
        y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    y_out = shallow_nn(x)

    with tf.variable_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out))

    optimiser = tf.train.AdamOptimizer(
        FLAGS.learning_rate, name="AdamOpt").minimize(cross_entropy)

    correct_prediction = tf.equal(
        tf.argmax(y, axis=1), tf.argmax(y_out, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_data, train_labels = get_data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.max_steps):

            permutation = np.random.permutation(len(train_data))
            shuffled_train_data = train_data[permutation]
            shuffled_train_labels = train_labels[permutation]
            train_data_batched = np.array_split(
                shuffled_train_data, FLAGS.batch_size)
            train_labels_batched = np.array_split(
                shuffled_train_labels, FLAGS.batch_size)

            for i, batch in enumerate(train_data_batched):
                step = epoch * len(train_data_batched) + i

                sess.run(optimiser, feed_dict={
                    x: batch, y: train_labels_batched[i]})

                if step % FLAGS.log_frequency == 0:
                    validation_accuracy = sess.run(
                        accuracy, feed_dict={x: batch, y: train_labels_batched[i]})
                    print('step %d, accuracy on validation batch: %g' %
                          (step, validation_accuracy))


if __name__ == '__main__':
    tf.app.run(main=main)
