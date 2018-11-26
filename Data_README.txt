Dataset:

We prepared the dataset for you stored in music_genres_dataset.pkl.

To load the dataset add the following lines in your code:

import pickle

with open('music_genres_dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)

Each set is a dictionary with 3 keys: 

'data': A list where each entry is an audio segment
'labels': A list where each entry is a 0-based integer label
'track_id': A list where each entry is a unique track id and all the audio segments belonging to the same track have the same id, useful for computing the maximum probability and  majority vote metrics.

---------------------------------------------------------------------------------------------------------
Librosa:

Librosa is a library for music and audio processing. You will use it for audio data augmentation, and to extract spectrograms (given in utils.py).

To install librosa, run in the terminal:

$ pip install -u librosa

----------------------------------------------------------------------------------------------------------
Data augmentation:

For data augmentation you will need the following two functions from Librosa:

- librosa.effects.pitch_shift(y, sr, n_steps, bins_per_octave=12), where sr is the sampling rate of audio, use sr=22050. n_steps is the number of semitones to lower/raise the audio signal.

- librosa.effects.time_stretch(y, rate), where rate is the 'multiplication factor' mentioned in the paper. This function modifies the input length. If length is shorter pad with zeros, and if longer truncate the audio to the original audio segment size.

The data augmentation should be done offline, and not during training, so the resulting augmented training set will be larger in size. To ensure that you did it correctly compare the size of your augmented dataset to that in Table 1. in the paper. 

------------------------------------------------------------------------------------------------------------- 

Spectrogram extraction:

To extract spectrogram for a sound segment, use melspectrogram(audio) from utils.py

-------------------------------------------------------------------------------------------------------------

Class mapping:

In classInd.txt you can find the mapping from the labels integers to the actual names of classes, useful for evaluation.




