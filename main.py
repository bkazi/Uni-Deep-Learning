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
tf.app.flags.DEFINE_integer('log-frequency', 100,
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

    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0001, scope=None)
    weights = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(
        l1_regularizer, weights)
    regularized_loss = cross_entropy + regularization_penalty

    optimiser = tf.train.AdamOptimizer(
        FLAGS.learning_rate, name="AdamOpt").minimize(regularized_loss)

    correct_prediction = tf.equal(
        tf.argmax(y, axis=1), tf.argmax(y_out, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_data, train_labels, test_data, test_labels = get_data()

    train_data_tf = tf.constant(train_data)
    train_labels_tf = tf.constant(train_labels)
    test_data_tf = tf.constant(test_data)
    test_labels_tf = tf.constant(test_labels)

    train_data_batch_op, train_labels_batch_op = tf.train.shuffle_batch(
        [train_data_tf, train_labels_tf],
        enqueue_many=True,
        batch_size=FLAGS.batch_size,
        capacity=len(train_data),
        min_after_dequeue=FLAGS.batch_size,
        allow_smaller_final_batch=True
    )
    test_data_batch_op, test_labels_batch_op = tf.train.shuffle_batch(
        [test_data_tf, test_labels_tf],
        enqueue_many=True,
        batch_size=FLAGS.batch_size,
        capacity=len(test_data),
        min_after_dequeue=FLAGS.batch_size,
        allow_smaller_final_batch=True
    )

    num_batches = int(len(train_data) / FLAGS.batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(FLAGS.max_steps):
            for i in range(num_batches):
                step = epoch * num_batches + i

                train_data_batch, train_labels_batch = sess.run(
                    [train_data_batch_op, train_labels_batch_op])
                sess.run(optimiser, feed_dict={
                    x: train_data_batch, y: train_labels_batch})

                if step % FLAGS.log_frequency == 0:
                    test_data_batch, test_labels_batch = sess.run(
                        [test_data_batch_op, test_labels_batch_op])
                    validation_accuracy = sess.run(
                        accuracy, feed_dict={x: test_data_batch, y: test_labels_batch})
                    print('step %d, accuracy on validation batch: %g' %
                          (step, validation_accuracy))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run(main=main)
