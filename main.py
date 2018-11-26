import pickle
import numpy as np
import tensorflow as tf

from utils import melspectrogram

with open('music_genres_dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)

train_set_data = map(lambda x: np.array(
    melspectrogram(x)), train_set["data"])
train_set_labels = np.eye(10)[train_set["labels"]]

print(train_set_labels)
