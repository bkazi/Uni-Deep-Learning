from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from functools import reduce

from utils import get_data, MusicGenreDataset
from shallow_nn import shallow_nn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('epochs', 3,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save_model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning_rate', 5e-5,
                          'Learning rate (default: %(default)d)')
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


run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))


def model(iterator):
    next_x, next_y = iterator.get_next()

    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
        y_out = shallow_nn(next_x)

    # Compute categorical loss
    with tf.variable_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_y, logits=y_out))

    # L1 regularise
    regularization_penalty = tf.losses.get_regularization_loss(
        scope=None, name='total_regularization_loss')
    regularized_loss = cross_entropy + regularization_penalty

    return regularized_loss


def calc_accuracy(iterator, nn):
    next_x, next_y = iterator.get_next()

    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
        y_out = nn(next_x)

    correct_prediction = tf.equal(
        tf.argmax(next_y, axis=1), tf.argmax(y_out, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

def evaluate(iterator, nn):
    x, y, i = iterator.get_next()

    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
        y_out = nn(x)

    return (x, y, y_out, i)

def np_softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def main(_):

    (train_set_data, train_set_labels, _, test_set_data,
     test_set_labels, test_set_track_ids) = get_data()

    features_placeholder = tf.placeholder(
        train_set_data.dtype, train_set_data.shape)
    labels_placeholder = tf.placeholder(
        train_set_labels.dtype, train_set_labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices(
        (features_placeholder, labels_placeholder))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(FLAGS.batch_size)
    train_iterator = dataset.make_initializable_iterator()

    test_features_placeholder = tf.placeholder(
        test_set_data.dtype, test_set_data.shape)
    test_labels_placeholder = tf.placeholder(
        test_set_labels.dtype, test_set_labels.shape)
    test_track_ids_placeholder = tf.placeholder(
        test_set_track_ids.dtype, test_set_track_ids.shape)

    dataset = tf.data.Dataset.from_tensor_slices(
        (test_features_placeholder, test_labels_placeholder))
    dataset = dataset.batch(FLAGS.batch_size)
    test_iterator = dataset.make_initializable_iterator()

    eval_dataset = tf.data.Dataset.from_tensor_slices((test_features_placeholder, test_labels_placeholder, test_track_ids_placeholder))
    eval_dataset = eval_dataset.batch(1)
    eval_iterator = eval_dataset.make_initializable_iterator()

    loss = model(train_iterator)

    # Adam Optimiser
    # default values match that in paper
    optimiser = tf.train.AdamOptimizer(
        FLAGS.learning_rate, name="AdamOpt").minimize(loss)

    validation_accuracy = calc_accuracy(test_iterator, shallow_nn)

    loss_summary = tf.summary.scalar('Loss', loss)
    acc_summary = tf.summary.scalar('Accuracy', validation_accuracy)

    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(
            run_log_dir + 'train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(
            run_log_dir+'_validate', sess.graph)

        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.epochs):
            sess.run(train_iterator.initializer, feed_dict={
                features_placeholder: train_set_data, labels_placeholder: train_set_labels})
            while True: 
                try:
                    # Run until all samples done
                    _, summary_str = sess.run([optimiser, loss_summary])
                except tf.errors.OutOfRangeError:
                    break

            summary_writer.add_summary(summary_str, epoch)

            sess.run(test_iterator.initializer, feed_dict={
                test_features_placeholder: test_set_data, test_labels_placeholder: test_set_labels})
            accuracies = []
            while True:
                try:
                    temp_acc, acc_summary_str = sess.run(
                        [validation_accuracy, acc_summary])
                    accuracies.append(temp_acc)
                except tf.errors.OutOfRangeError:
                    break

            summary_writer_validation.add_summary(acc_summary_str, epoch)

            print("Validation accuracy on epoch " +
                  str(epoch) + ": ", np.mean(accuracies))

        evaluator = evaluate(eval_iterator, shallow_nn)
        sess.run(eval_iterator.initializer, feed_dict={test_features_placeholder: test_set_data, test_labels_placeholder: test_set_labels, test_track_ids_placeholder: test_set_track_ids})

        results = []

        while True:
            try:
                evaluated = sess.run(evaluator)
                results.append(evaluated)
            except tf.errors.OutOfRangeError:
                break

        raw_probability = []
        maximum_probability = []
        majority_vote = []

        track_truth = {}
        track_softmax = {}
        track_predictions = {}

        for result in results:
            y = result[1].flatten()
            y_out = result[2].flatten()
            y_out_prediction = np.eye(FLAGS.num_classes)[np.argmax(y_out)]
            y_out_softmax = np_softmax(y_out)
            i = result[3][0]

            track_truth[i] = y
            track_softmax[i] = track_softmax.get(i, []) + [y_out_softmax] # Bug here sometimes plus wants to be numeric
            track_predictions[i] = track_predictions.get(i, []) + [y_out_prediction]

            raw_probability.append(int(np.array_equal(y, y_out_prediction)))

        for i in track_softmax:
            truth = track_truth[i]
            softmaxs = track_softmax[i]
            predictions = track_predictions[i]

            print("i: ", i)
            print("Softmaxs: ", softmaxs)

            track_softmax = np_softmax(reduce((lambda x, y: np.add(x, y)), softmaxs))
            maximum_probability.append(int(np.array_equal(truth, np.eye(FLAGS.num_classes)[np.argmax(track_softmax)])))

            track_prediction = np.eye(FLAGS.num_classes)[np.argmax(reduce((lambda x, y: np.add(x, y)), predictions))]
            majority_vote.append(int(np.array_equal(truth, track_prediction)))
            
        
        print("-----===== Summary =====-----")
        print("Raw Probability: ", np.mean(raw_probability))
        print("Maximum Probability: ", np.mean(maximum_probability))
        print("Majority Vote: ", np.mean(majority_vote))


if __name__ == '__main__':
    tf.app.run(main=main)
