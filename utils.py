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


def get_data():
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)

    train_set_data = map(lambda x: np.array(
        melspectrogram(x)), train_set["data"])
    train_set_labels = np.eye(FLAGS.num_classes)[train_set["labels"]]
    return (train_set_data, train_set_labels)
