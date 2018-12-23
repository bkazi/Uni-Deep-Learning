import pickle
import librosa
import numpy as np
import tensorflow as tf
import itertools
import random


FLAGS = tf.app.flags.FLAGS
SAMPLING_RATE = 22050


def melspectrogram(audio):
    spec = librosa.stft(
        audio,
        n_fft=512,
        window="hann",
        hop_length=256,
        win_length=512,
        pad_mode="constant",
    )
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)


def tf_melspectogram(audio):
    spec = tf.contrib.signal.stft(
        audio, frame_length=512, frame_step=256, fft_length=512, pad_end=True
    )
    mag_spec = tf.abs(spec)
    num_spectrogram_bins = mag_spec.shape[-1].value
    num_mel_bins, lower_edge_hertz, upper_edge_hertz = 80, 0, (SAMPLING_RATE / 2)
    mel_basis = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        SAMPLING_RATE,
        lower_edge_hertz,
        upper_edge_hertz,
    )
    mel_spec = tf.tensordot(mag_spec, mel_basis, 1)
    mel_spec.set_shape(mag_spec.shape[:-1].concatenate(mel_basis.shape[-1:]))
    mel_spec = tf.expand_dims(mel_spec, -1)
    return tf.log(mel_spec + 1e-6)


def preprocess_py_func(features, label):
    transformed = melspectrogram(features)[:, :, np.newaxis]
    transformed = transformed.astype("float32")
    return transformed, label


def augmentFunctions(features, params):
    timeStretched = librosa.effects.time_stretch(features, params[0])
    pitchShifted = librosa.effects.pitch_shift(timeStretched, SAMPLING_RATE, params[1])

    return pitchShifted


def dataAugmentation(features):
    timeStretchValues = [1.2, 1.5, 0.5, 0.2]
    pitchShifting = [-2, -5, 2, 5]

    combinations = itertools.product(timeStretchValues, pitchShifting)
    augmentedData = map(lambda x: augmentFunctions(features, x), combinations)
    return augmentedData


def augmentData(trainingSetData, trainingSetLabels):
    augmentedData = trainingSetData
    augmentedLabels = trainingSetLabels
    newData = []
    newLabels = []
    print("Augmenting Data!")
    for ind, data in enumerate(augmentData):
        randNum = np.random.random_sample()
        if randNum <= 0.20:  # 1 in 5 chance
            addedData = dataAugmentation(data)
            newData = np.append(newData, addedData)
            newLabels = np.append(
                newLabels, np.repeat([augmentedLabels[ind]], len(addedData))
            )
            # repeat the label

    augmentedData["data"] = augmentedData["data"] + newData
    augmentedData["labels"] = augmentedData["labels"] + newLabels

    return augmentedData


def get_data():
    with open("music_genres_dataset.pkl", "rb") as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)

    # print(train_set)
    # print(type(train_set))
    train_set = augmentData(train_set["data"], train_set["labels"])

    train_set_data = train_set["data"]
    train_set_labels = train_set["labels"]
    test_set_data = test_set["data"]
    test_set_labels = test_set["labels"]
    return (train_set_data, train_set_labels, test_set_data, test_set_labels)


"""
Data Augmentation writing up:
  # # 1 in 5 chance
    # randNum = tf.random.uniform([1])
    # toAugment = tf.less_equal(randNum, [0.20])
    # augFeatures = tf.cond(toAugment[0], lambda: tf_dataAug(features), lambda: features)
    # augFeatures = tf.cond(toAugment[0], lambda: dataAugmentation(features), lambda: features)
    # augFeatures = tf_dataAug(features)
    # augFeatures.set_shape([None, 80, 1])
    # return augFeatures, label



"""

