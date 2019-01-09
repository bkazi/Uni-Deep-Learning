from utils import melspectrogram
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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


def make_prediction(prob):
    return np.eye(
        FLAGS.num_classes)[np.argmax(prob)]


def find_interesting(track_id, track_inputs, track_truth, track_raw_probability, track_maximum_probability, track_majority_vote):
    fig = plt.figure()

    maximum_probability_correct = np.array_equal(
        track_truth, make_prediction(track_maximum_probability))
    majority_vote_correct = np.array_equal(
        track_truth, make_prediction(track_majority_vote))

    truth_index = np.argmax(track_truth)

    for idx, raw_probability in enumerate(track_raw_probability):
        raw_probability_correct = np.array_equal(
            track_truth, make_prediction(raw_probability))

        if not raw_probability_correct and maximum_probability_correct or majority_vote_correct:
            raster = melspectrogram(
                track_inputs[idx])
            plt.imshow(raster)
            fig.savefig("specimages/caseA-{}-{}.png".format(track_id, idx))
            plt.clf()

        truth_confidence = raw_probability[truth_index]
        sorted_confidences = np.sort(raw_probability)
        truth_rank = 0
        for confidence in sorted_confidences:
            if confidence == truth_confidence:
                break
            else:
                truth_rank += 1

        if truth_rank == 1 or truth_rank == 2:
            raster = melspectrogram(
                track_inputs[idx])
            plt.imshow(raster)
            fig.savefig("specimages/caseB-{}-{}.png".format(track_id, idx))
            plt.clf()

        if truth_rank > 2:
            raster = melspectrogram(
                track_inputs[idx])
            plt.imshow(raster)
            fig.savefig("specimages/caseC-{}-{}.png".format(track_id, idx))
            plt.clf()

    plt.close(fig)


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
        track_inputs = track_xs[track_id]
        track_truths = track_ys[track_id]
        track_truth = track_truths[0]
        track_raw_probability = track_y_outs[track_id]

        track_raw_probability_predictions = map(
            lambda x: make_prediction(x), track_raw_probability)
        raw_probability.extend((np.argmax(
            track_raw_probability_predictions, axis=1) == np.argmax(track_truths, axis=1)).astype(int))

        track_maximum_probability = reduce(
            (lambda x, y: np.add(x, y)), track_raw_probability)
        track_maximum_probability_prediction = make_prediction(
            track_maximum_probability)
        maximum_probability.append(int(np.array_equal(
            track_truth, track_maximum_probability_prediction)))

        track_majority_vote = reduce(
            (lambda x, y: np.add(x, y)), track_raw_probability_predictions)
        track_majority_vote_prediction = make_prediction(track_majority_vote)
        majority_vote.append(
            int(np.array_equal(track_truth, track_majority_vote_prediction)))

        find_interesting(track_id, track_inputs, track_truth, track_raw_probability,
                         track_maximum_probability, track_majority_vote)

    return (np.mean(raw_probability), np.mean(maximum_probability), np.mean(majority_vote))
