import pickle

with open('music_genres_dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)

print(train_set)
