import pickle
import librosa
import numpy as np
import tensorflow as tf
import itertools
import random
from multiprocessing import Pool
import time


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
    num_mel_bins, lower_edge_hertz, upper_edge_hertz = 80, 0, (
        SAMPLING_RATE / 2)
    mel_basis = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, SAMPLING_RATE, lower_edge_hertz, upper_edge_hertz)
    mel_basis = tf.transpose(mel_basis)
    mel_spec = tf.map_fn(lambda x: tf.matmul(
        mel_basis, x), tf.transpose(mag_spec, perm=[0, 2, 1]))
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


def pad(x, length):
    x_len = x.shape[0]
    if x_len > length:
        return x[:length]
    else:
        return np.hstack((x, np.zeros(length - x_len)))


def augmentInner(sample_param):
    features, params = sample_param
    return pad(augmentFunctions(features, params), np.shape(features)[0])


def dataAugmentationFeatures(samples, pool):
    timeStretchValues = [1.2, 1.5, 0.5, 0.2]
    pitchShifting = [-2, -5, 2, 5]
    param_combinations = itertools.product(timeStretchValues, pitchShifting)
    samples_param_combinations = itertools.product(samples, param_combinations)

    return pool.map(augmentInner, samples_param_combinations)


num_samples = 3
num_segments_per_track = 15
num_augments = 16


def dataAugmentationLabels(labels):
    return np.repeat(labels, num_augments)


def augmentDataFunc(trainingSetData, trainingSetLabels):
    print("Augmenting Data!")

    sampleIdx = [np.random.randint(i * num_segments_per_track, (i + 1) * num_segments_per_track, size=num_samples)
                 for i in range(len(trainingSetData) / num_segments_per_track)]
    sampleIdx = np.array(sampleIdx).flatten()
    samplesToAugment = trainingSetData[sampleIdx]
    labelsToRepeat = trainingSetLabels[sampleIdx]

    p = Pool(FLAGS.num_parallel_calls)
    newData = dataAugmentationFeatures(samplesToAugment, p)
    newLabels = p.map(dataAugmentationLabels, labelsToRepeat)
    p.close()
    p.join()

    newData = np.array(newData)
    newLabels = np.array(newLabels).flatten()

    train_len = len(trainingSetData)
    new_len = len(newData)

    trainingData = np.vstack(
        (trainingSetData[train_len/2:], newData[:new_len/2], trainingSetData[:train_len/2], newData[new_len/2:]))
    trainingLabels = np.hstack(
        (trainingSetLabels[train_len/2:], newLabels[:new_len/2], trainingSetLabels[:train_len/2], newLabels[new_len/2:]))

    return trainingData, trainingLabels


def get_data():
    print("Reading data")
    start = time.time()
    with open("music_genres_dataset.pkl", "rb") as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
    end = time.time()
    print("Time time to read dataset: {:.2f}s".format(end - start))

    train_set_data = np.array(train_set["data"])
    train_set_labels = np.array(train_set["labels"])

    if (FLAGS.augment == 1):
        start = time.time()
        train_set_data, train_set_labels = augmentDataFunc(
            train_set_data, train_set_labels)
        end = time.time()
        print("Time to augment dataset: {:.2f}m".format((end - start) / 60.0))

    print(
        "Training Set Data: {0}, training set labels: {1}".format(
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
