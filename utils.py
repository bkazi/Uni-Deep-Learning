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

    train_set_data = np.array(map(lambda x: melspectrogram(x)[
                              :, :, np.newaxis], train_set["data"]))
    train_set_labels = np.eye(FLAGS.num_classes)[train_set["labels"]]
    train_set_track_ids = train_set["track_id"]

    test_set_data = np.array(map(lambda x: melspectrogram(x)[
                             :, :, np.newaxis], test_set["data"]))
    test_set_labels = np.eye(FLAGS.num_classes)[test_set["labels"]]
    test_set_track_ids = test_set["track_id"]
    return (train_set_data, train_set_labels, train_set_track_ids, test_set_data, test_set_labels, test_set_track_ids)


class MusicGenreDataset:
    def __init__(self):
        with open('music_genres_dataset.pkl', 'rb') as f:
            self.train_set = pickle.load(f)
            self.test_set = pickle.load(f)

        self.train_data_size = len(self.train_set["data"])
        self.test_data_size = len(self.test_set["data"])

        train_data = np.array(map(lambda x: melspectrogram(x)[
                              :, :, np.newaxis], self.train_set["data"]))
        train_labels = np.eye(FLAGS.num_classes)[self.train_set["labels"]]
        test_data = np.array(map(lambda x: melspectrogram(x)[
            :, :, np.newaxis], self.test_set["data"]))
        test_labels = np.eye(FLAGS.num_classes)[self.test_set["labels"]]

        train_data_tf = tf.constant(train_data)
        train_labels_tf = tf.constant(train_labels)
        test_data_tf = tf.constant(test_data)
        test_labels_tf = tf.constant(test_labels)

        NUM_THREADS = 4

        self.train_data_batch_op, self.train_labels_batch_op = tf.train.shuffle_batch(
            [train_data_tf, train_labels_tf],
            enqueue_many=True,
            batch_size=FLAGS.batch_size,
            capacity=len(train_data) / 2,
            min_after_dequeue=FLAGS.batch_size,
            allow_smaller_final_batch=True,
            num_threads=NUM_THREADS
        )

        self.test_data_batch_op, self.test_labels_batch_op = tf.train.shuffle_batch(
            [test_data_tf, test_labels_tf],
            enqueue_many=True,
            batch_size=FLAGS.batch_size,
            capacity=len(test_data) / 2,
            min_after_dequeue=FLAGS.batch_size,
            allow_smaller_final_batch=True,
            num_threads=NUM_THREADS
        )

    def getTrainBatch(self, sess):
        return sess.run([self.train_data_batch_op, self.train_labels_batch_op])

    def getTestBatch(self, sess):
        return sess.run([self.test_data_batch_op, self.test_labels_batch_op])
