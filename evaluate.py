import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

'''
    Softmax for just numpy
'''
def np_softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
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
    raw_probability = []
    maximum_probability = []
    majority_vote = []

    track_truth = {}
    track_softmaxs = {}
    track_predictions = {}

    for result in results:
        y = result[1].flatten()
        y_out = result[2].flatten()
        y_out_prediction = np.eye(FLAGS.num_classes)[np.argmax(y_out)]
        y_out_softmax = np_softmax(y_out)
        i = result[3][0]

        track_truth[i] = y
        track_softmaxs[i] = f_extend(track_softmaxs.get(i, []), [y_out_softmax]) # Bug here sometimes plus wants to be numeric
        track_predictions[i] = f_extend(track_predictions.get(i, []), [y_out_prediction])

        raw_probability.append(int(np.array_equal(y, y_out_prediction)))

    for i in track_truth:
        truth = track_truth[i]
        softmaxs = track_softmaxs[i]
        predictions = track_predictions[i]

        track_softmax = np_softmax(reduce((lambda x, y: np.add(x, y)), softmaxs))
        maximum_probability.append(int(np.array_equal(truth, np.eye(FLAGS.num_classes)[np.argmax(track_softmax)])))

        track_prediction = np.eye(FLAGS.num_classes)[np.argmax(reduce((lambda x, y: np.add(x, y)), predictions))]
        majority_vote.append(int(np.array_equal(truth, track_prediction)))
        
    
    print("-----===== Summary =====-----")
    print("Raw Probability: ", np.mean(raw_probability))
    print("Maximum Probability: ", np.mean(maximum_probability))
    print("Majority Vote: ", np.mean(majority_vote))