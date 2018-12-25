from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from evaluate import evaluate
from utils import get_data, tf_melspectogram
from shallow_nn import shallow_nn
from deep_nn import deep_nn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('epochs', 100,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('network', 0,
                            'Type of network to use, 0 for shallow, 1 for deep. (default: %(default)d)')
tf.app.flags.DEFINE_integer('bn', 0,
                            'Turn batch norm on or off, 0 for off, 1 for improvements on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('decay', 0,
                            'Turn decaying learning rate on or off. (default: %(default)d')
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('num_parallel_calls', 1,
                            'Number of cpu cores to use to preprocess data')
tf.app.flags.DEFINE_integer('save_model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float(
    'learning_rate', 5e-5, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'input_width', 80, 'Input width (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'input_height', 80, 'Input height (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'input_channels', 1, 'Input channels (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'num_classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


run_log_dir = os.path.join(FLAGS.log_dir, 'exp_lr_{learning_rate}_decay_{decay}_bs_{batch_size}_e_{epochs}_{network}_bn_{bn}'.format(
    learning_rate=FLAGS.learning_rate, decay={FLAGS.decay}, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, network='shallow' if (FLAGS.network == 0) else 'deep', bn=FLAGS.bn))


def model(iterator, is_training, nn):
    next_x, next_y = iterator.get_next()

    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
        y_out, img_summary = nn(next_x, is_training)

    # Compute categorical loss
    with tf.variable_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_y, logits=y_out))

    # L1 regularise
    regularization_penalty = tf.losses.get_regularization_loss(
        name='total_regularization_loss')
    regularized_loss = cross_entropy + regularization_penalty

    return regularized_loss, img_summary


def calc_accuracy(iterator, is_training, nn):
    next_x, next_y = iterator.get_next()

    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
        y_out, _ = nn(next_x, is_training)

    accuracy, accuracy_op = tf.metrics.accuracy(
        tf.argmax(next_y, axis=1), tf.argmax(y_out, axis=1), name="accuracy")

    return accuracy, accuracy_op


def accumulate_results(iterator, is_training, nn):
    x, y, i = iterator.get_next()

    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
        y_out, _ = nn(x, is_training)

    return (x, y, y_out, i)


def _preprocess(features, label):
    label = tf.one_hot(indices=label, depth=FLAGS.num_classes, dtype=tf.uint8)
    return features, label


def main(_):

    (train_set_data, train_set_labels, _, test_set_data,
     test_set_labels, test_set_track_ids) = get_data()

    is_training_placeholder = tf.placeholder_with_default(False, shape=())

    features_placeholder = tf.placeholder(
        tf.float32, (None, np.shape(train_set_data)[1]))
    labels_placeholder = tf.placeholder(
        tf.uint8, (None))
    track_ids_placeholder = tf.placeholder(tf.uint8, (None))

    dataset = tf.data.Dataset.from_tensor_slices(
        (features_placeholder, labels_placeholder))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(_preprocess)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    train_iterator = dataset.make_initializable_iterator()
    test_iterator = dataset.make_initializable_iterator()

    eval_dataset = tf.data.Dataset.from_tensor_slices(
        (features_placeholder, labels_placeholder, track_ids_placeholder))
    eval_dataset = eval_dataset.map(lambda features, label, track_id: (
        features, tf.one_hot(indices=label, depth=FLAGS.num_classes, dtype=tf.uint8), track_id))
    eval_dataset = eval_dataset.batch(1)
    eval_iterator = eval_dataset.make_initializable_iterator()

    nn = shallow_nn if (FLAGS.network == 0) else deep_nn

    loss, img_summary = model(
        train_iterator, is_training_placeholder, nn)

    global_step = tf.Variable(0, trainable=False)
    if (FLAGS.decay):
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, 1000, 1e-5)
    else:
        learning_rate = FLAGS.learning_rate
    # Adam Optimiser
    # default values match that in paper
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimiser = tf.train.AdamOptimizer(
            learning_rate, name="AdamOpt").minimize(loss, global_step=global_step)

    validation_accuracy, acc_op = calc_accuracy(
        test_iterator, is_training_placeholder, nn)

    loss_summary = tf.summary.scalar('Loss', loss)
    acc_summary = tf.summary.scalar('Accuracy', validation_accuracy)

    training_summary = tf.summary.merge([img_summary, loss_summary])
    validation_summary = tf.summary.merge([acc_summary])

    # Isolate the variables stored behind the scenes by the metric operation
    running_vars = tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")

    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(
        var_list=running_vars)

    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(
            run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(
            run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.epochs):
            sess.run(running_vars_initializer)
            sess.run(train_iterator.initializer, feed_dict={
                features_placeholder: train_set_data, labels_placeholder: train_set_labels})

            # Run until all samples done
            while True:
                try:
                    _, summary_str = sess.run([optimiser, training_summary], feed_dict={
                                              is_training_placeholder: True})
                except tf.errors.OutOfRangeError:
                    break

            summary_writer.add_summary(summary_str, epoch)

            sess.run(test_iterator.initializer, feed_dict={
                features_placeholder: test_set_data, labels_placeholder: test_set_labels})
            while True:
                try:
                    acc, acc_summary_str = sess.run(
                        [acc_op, validation_summary])
                except tf.errors.OutOfRangeError:
                    break

            summary_writer_validation.add_summary(acc_summary_str, epoch)
            print("Validation accuracy after epoch " +
                  str(epoch) + ": ", acc)

        evaluator = accumulate_results(
            eval_iterator, is_training_placeholder, nn)
        sess.run(eval_iterator.initializer, feed_dict={
                 features_placeholder: test_set_data, labels_placeholder: test_set_labels, track_ids_placeholder: test_set_track_ids})

        results = []

        while True:
            try:
                evaluated = sess.run(evaluator)
                results.append(evaluated)
            except tf.errors.OutOfRangeError:
                break

        raw_probability, maximum_probability, majority_vote = evaluate(results)

        print("-----===== Summary =====-----")
        print("Raw Probability: ", raw_probability)
        print("Maximum Probability: ", maximum_probability)
        print("Majority Vote: ", majority_vote)


if __name__ == '__main__':
    tf.app.run(main=main)
