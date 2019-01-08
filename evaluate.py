import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

'''
    Softmax for just numpy
'''


def np_softmax(w):
    # print("w")
    # print(w)
    e = np.exp(np.array(w) - np.max(w))
    # print("e")
    # print(e)
    dist = e / np.sum(e)
    # print("dist")
    # print(dist)
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
        truths = track_ys[track_id]
        truth = truths[0]
        softmaxs = track_y_outs[track_id]
        predictions = map(lambda x: np.eye(FLAGS.num_classes)
                          [np.argmax(x)], softmaxs)
        corrects = np.argmax(predictions, axis=1) == np.argmax(truths, axis=1)

        raw_probability.extend(corrects.astype(int))

        track_softmax = np_softmax(
            reduce((lambda x, y: np.add(x, y)), softmaxs))
        maximum_probability.append(int(np.array_equal(
            truth, np.eye(FLAGS.num_classes)[np.argmax(track_softmax)])))

        track_prediction = np.eye(FLAGS.num_classes)[np.argmax(
            reduce((lambda x, y: np.add(x, y)), predictions))]
        majority_vote.append(int(np.array_equal(truth, track_prediction)))

    return (np.mean(raw_probability), np.mean(maximum_probability), np.mean(majority_vote))
