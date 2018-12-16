import pickle
import librosa
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann',
                        hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)


def tf_melspectogram(audio):
    sample_rate = 22050
    spec = tf.contrib.signal.stft(
        audio, frame_length=512, frame_step=256, fft_length=512, pad_end=True)
    mag_spec = tf.abs(spec)
    num_spectrogram_bins = mag_spec.shape[-1].value
    num_mel_bins, lower_edge_hertz, upper_edge_hertz = 80, 0, sample_rate/2
    mel_basis = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spec = tf.tensordot(mag_spec, mel_basis, 1)
    mel_spec.set_shape(
        mag_spec.shape[:-1].concatenate(mel_basis.shape[-1:]))
    mel_spec = tf.expand_dims(mel_spec, -1)
    return tf.log(mel_spec + 1e-6)


def preprocess_py_func(features, label):
    transformed = melspectrogram(features)[:, :, np.newaxis]
    transformed = transformed.astype('float32')
    return transformed, label


def get_data():
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)

    train_set_data = train_set["data"]
    train_set_labels = train_set["labels"]
    test_set_data = test_set["data"]
    test_set_labels = test_set["labels"]
    return (train_set_data, train_set_labels, test_set_data, test_set_labels)
