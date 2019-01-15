import pickle
import math
import os
from scipy.io.wavfile import write

with open('music_genres_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    for idx, segment in enumerate(data['data']):
        track = int(math.floor(idx/15))
        part = idx % 15

        if not os.path.exists('tracks/{}'.format(track)):
            os.makedirs('tracks/{}'.format(track))

        write('tracks/{}/{}.wav'.format(track, part), 22050, segment)
