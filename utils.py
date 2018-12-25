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
    pitchShifted = np.array(
        librosa.effects.pitch_shift(timeStretched, SAMPLING_RATE, params[1])
    )

    return pitchShifted


def dataAugmentation(features):
    features = np.array(features)
    timeStretchValues = [1.2, 1.5, 0.5, 0.2]
    pitchShifting = [-2, -5, 2, 5]
    pad = (
        lambda a, i: a[0:i]
        if a.shape[0] > i
        else np.hstack((a, np.zeros(i - a.shape[0])))
    )

    combinations = itertools.product(timeStretchValues, pitchShifting)
    augmentedData = np.array(
        map(
            lambda x: pad(augmentFunctions(features, x), features.shape[0]),
            combinations,
        )
    )

    return augmentedData


def augmentDataFunc(trainingSetData, trainingSetLabels):

    newData = np.copy(trainingSetData)
    newLabels = np.copy(trainingSetLabels)

    print("\n \nAugmenting Data! \n \n")
    pad = (
        lambda a, i: a[0:i]
        if a.shape[0] > i
        else np.hstack((a, np.zeros(i - a.shape[0])))
    )

    for ind, segment in enumerate(trainingSetData):
        randNum = np.random.random_sample()
        if randNum <= 0.20:  # 1 in 5 chance
            addedData = np.array(dataAugmentation(segment))
            newData = np.append(newData, addedData, axis=0)
            newLabels = np.append(
                newLabels, np.repeat([trainingSetLabels[ind]], len(addedData)), axis=0
            )

    print(
        "BEFORE APPENDING AugmentedData : {0}, augmentedLabels :{1} ".format(
            trainingSetData.shape, trainingSetLabels.shape
        )
    )

    trainingData = newData
    trainingLabels = newLabels

    print(
        "AFTER APPENDING AugmentedData : {0}, augmentedLabels :{1} ".format(
            trainingData.shape, trainingLabels.shape
        )
    )

    return trainingData, trainingLabels


def get_data():
    with open("music_genres_dataset.pkl", "rb") as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)

    print(
        "BEFORE : Training Set Data : {0}, training set labels : {1}".format(
            len(train_set["data"]), len(train_set["labels"])
        )
    )

    train_set_data = train_set["data"]
    train_set_labels = train_set["labels"]

    train_set_data = np.array(train_set_data)
    train_set_labels = np.array(train_set_labels)

    train_set_data, train_set_labels = augmentDataFunc(train_set_data, train_set_labels)

    print(
        "AFTER : Training Set Data : {0}, training set labels : {1}".format(
            len(train_set_data), len(train_set_labels)
        )
    )

    train_set_track_ids = train_set["track_id"]
    test_set_data = test_set["data"]
    test_set_labels = test_set["labels"]
    test_set_track_ids = test_set["track_id"]
    return (
        train_set_data,
        train_set_labels,
        train_set_track_ids,
        test_set_data,
        test_set_labels,
        test_set_track_ids,
    )

