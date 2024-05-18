import pandas as pd
from sklearn.model_selection import train_test_split

ds = pd.read_csv('data/preprocessed.csv')

X = ds.drop('playlist_genre_encoded', axis=1)
Y = ds['playlist_genre_encoded']

train, test = train_test_split(ds, train_size=0.7, test_size=0.3)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/new_input.csv", index=False)