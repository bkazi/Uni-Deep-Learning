import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

'''
    Softmax for just numpy
'''


def np_softmax(w):
    e = np.exp(np.array(w) - np.max(w))
    dist = e / np.sum(e)
    return dist


'''
    Functional list extend - WHY DO LANGUAGES INSIST ON MUTATION
'''


def f_extend(list1, list2):
    newlist = list(list1)
    newlist.extend(list2)
    return newlist


def evaluate(results):
    track_xs = {}
    track_ys = {}
    track_y_outs = {}

    for result in results:
        x = result[0].flatten()
        y = result[1].flatten()
        y_out = np_softmax(result[2].flatten())
        i = result[3][0]

        track_xs[i] = f_extend(track_xs.get(i, []), [x])
        track_ys[i] = f_extend(track_ys.get(i, []), [y])
        track_y_outs[i] = f_extend(track_y_outs.get(i, []), [y_out])

    raw_probability = []
    maximum_probability = []
    majority_vote = []

    for track_id in track_xs:
        track_truths = track_ys[track_id]
        track_truth = track_truths[0]
        track_probabilities = track_y_outs[track_id]

        track_raw_probability = map(lambda x: np.eye(FLAGS.num_classes)
                                    [np.argmax(x)], track_probabilities)
        raw_probability.extend(
            (np.argmax(track_raw_probability, axis=1) == np.argmax(track_truths, axis=1)).astype(int))

        track_maximum_probability = np.eye(FLAGS.num_classes)[
            np.argmax(
                reduce((lambda x, y: np.add(x, y)), track_probabilities))]
        maximum_probability.append(int(np.array_equal(
            track_truth, track_maximum_probability)))

        track_majority_vote = np.eye(FLAGS.num_classes)[np.argmax(
            reduce((lambda x, y: np.add(x, y)), track_raw_probability))]
        majority_vote.append(
            int(np.array_equal(track_truth, track_majority_vote)))

    return (np.mean(raw_probability), np.mean(maximum_probability), np.mean(majority_vote))
