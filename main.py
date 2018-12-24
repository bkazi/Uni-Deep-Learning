from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from functools import reduce

from utils import preprocess_py_func, get_data, tf_melspectogram, dataAugmentation
from deep_nn import shallow_nn
from evaluate import evaluate

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    "epochs", 100, "Number of mini-batches to train on. (default: %(default)d)"
)

tf.app.flags.DEFINE_integer(
    "num_parallel_calls", 1, "Number of cpu cores to use to preprocess data"
)
tf.app.flags.DEFINE_integer(
    "save_model", 1000, "Number of steps between model saves (default: %(default)d)"
)

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer(
    "batch_size", 16, "Number of examples per mini-batch (default: %(default)d)"
)
tf.app.flags.DEFINE_float("learning_rate", 5e-5, "Learning rate (default: %(default)d)")
tf.app.flags.DEFINE_integer("input_width", 80, "Input width (default: %(default)d)")
tf.app.flags.DEFINE_integer("input_height", 80, "Input height (default: %(default)d)")
tf.app.flags.DEFINE_integer(
    "input_channels", 1, "Input channels (default: %(default)d)"
)
tf.app.flags.DEFINE_integer(
    "num_classes", 10, "Number of classes (default: %(default)d)"
)
tf.app.flags.DEFINE_string(
    "log_dir",
    "{cwd}/logs/".format(cwd=os.getcwd()),
    "Directory where to write event logs and checkpoint. (default: %(default)s)",
)


run_log_dir = os.path.join(
    FLAGS.log_dir,
    "exp_bs_{bs}_lr_{lr}".format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate),
)


def model(iterator, is_training, nn):
    next_x, next_y = iterator.get_next()

    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE):
        y_out = nn(next_x, is_training)

    # Compute categorical loss
    with tf.variable_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_y, logits=y_out)
        )

    # L1 regularise
    regularization_penalty = tf.losses.get_regularization_loss(
        name="total_regularization_loss"
    )
    regularized_loss = cross_entropy + regularization_penalty

    return regularized_loss


def calc_accuracy(iterator, is_training, nn):
    next_x, next_y = iterator.get_next()

    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE):
        y_out = nn(next_x, is_training)

    correct_prediction = tf.equal(tf.argmax(next_y, axis=1), tf.argmax(y_out, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def accumulate_results(iterator, is_training, nn):
    x, y, i = iterator.get_next()

    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE):
        y_out = nn(x, is_training)

    return (x, y, y_out, i)


def _preprocess(features, label):
    label = tf.one_hot(indices=label, depth=FLAGS.num_classes, dtype=tf.uint8)
    return features, label


def main(_):

    (
        train_set_data,
        train_set_labels,
        _,
        test_set_data,
        test_set_labels,
        test_set_track_ids,
    ) = get_data()

    is_training_placeholder = tf.placeholder_with_default(False, shape=())

    features_placeholder = tf.placeholder(
        tf.float32, (None, np.shape(train_set_data)[1])
    )
    labels_placeholder = tf.placeholder(tf.uint8, (None))
    track_ids_placeholder = tf.placeholder(tf.uint8, (None))

    dataset = tf.data.Dataset.from_tensor_slices(
        (features_placeholder, labels_placeholder)
    )
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(_preprocess)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    train_iterator = dataset.make_initializable_iterator()
    test_iterator = dataset.make_initializable_iterator()

    eval_dataset = tf.data.Dataset.from_tensor_slices(
        (features_placeholder, labels_placeholder, track_ids_placeholder)
    )
    eval_dataset = eval_dataset.map(
        lambda features, label, track_id: (
            features,
            tf.one_hot(indices=label, depth=FLAGS.num_classes, dtype=tf.uint8),
            track_id,
        )
    )
    eval_dataset = eval_dataset.batch(1)
    eval_iterator = eval_dataset.make_initializable_iterator()

    loss = model(train_iterator, is_training_placeholder, shallow_nn)

    # Adam Optimiser
    # default values match that in paper
    optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate, name="AdamOpt").minimize(
        loss
    )

    validation_accuracy = calc_accuracy(
        test_iterator, is_training_placeholder, shallow_nn
    )

    loss_summary = tf.summary.scalar("Loss", loss)
    acc_summary = tf.summary.scalar("Accuracy", validation_accuracy)

    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        summary_writer_validation = tf.summary.FileWriter(
            run_log_dir + "_validate", sess.graph
        )

        sess.run(tf.global_variables_initializer())

        num_train_batches = round(len(train_set_data) / FLAGS.batch_size)
        num_test_batches = round(len(test_set_data) / FLAGS.batch_size)
        for epoch in range(FLAGS.epochs):
            sess.run(
                train_iterator.initializer,
                feed_dict={
                    features_placeholder: train_set_data,
                    labels_placeholder: train_set_labels,
                },
            )

            batch_counter = 0
            # Run until all samples done
            while True:
                try:
                    _, summary_str = sess.run(
                        [optimiser, loss_summary],
                        feed_dict={is_training_placeholder: True},
                    )
                    summary_writer.add_summary(
                        summary_str, epoch * num_train_batches + batch_counter
                    )
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break

            sess.run(
                test_iterator.initializer,
                feed_dict={
                    features_placeholder: test_set_data,
                    labels_placeholder: test_set_labels,
                },
            )
            accuracies = []
            batch_counter = 0
            while True:
                try:
                    temp_acc, acc_summary_str = sess.run(
                        [validation_accuracy, acc_summary]
                    )
                    summary_writer_validation.add_summary(
                        acc_summary_str, epoch * num_test_batches + batch_counter
                    )
                    accuracies.append(temp_acc)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break

            print(
                "Validation accuracy after epoch " + str(epoch) + ": ",
                np.mean(accuracies),
            )

        evaluator = accumulate_results(
            eval_iterator, is_training_placeholder, shallow_nn
        )
        sess.run(
            eval_iterator.initializer,
            feed_dict={
                features_placeholder: test_set_data,
                labels_placeholder: test_set_labels,
                track_ids_placeholder: test_set_track_ids,
            },
        )

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


if __name__ == "__main__":
    tf.app.run(main=main)
