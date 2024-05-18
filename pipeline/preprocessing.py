import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def missing_values(ds):
    ds = ds.dropna()

def milliseconds_to_seconds(val):
    return val/1000

def seconds_to_minutes(val):
    return val/60

def adding_features(ds):
    
    #'year' column
    date_formats = ['%Y', '%d/%m/%Y', '%Y-%m', '%m/%d/%Y', '%Y-%m', '%m/%Y', '%Y-%m-%d', '%m/%Y', '%Y-%m-%d']
    years = []

    for release_date in ds['track_album_release_date']:
        year_found = False

        for date_format in date_formats:
            try:
                year = pd.to_datetime(release_date, format=date_format).year
                ds['year'] = year
                year_found = True
                years.append(year)
                #print(t.year)
                break
            except ValueError:
                pass 
        if not year_found:
            years.append(None)

    ds['year'] = years

    #song duration in minutes and seconds
    ds['duration_s'] = ds['duration_ms'].apply(milliseconds_to_seconds)
    ds['duration_m'] = ds['duration_s'].apply(seconds_to_minutes)

def encoding(ds):
    ds.info()
    cols_to_encode = ['track_id', 'track_name', 'track_artist', 'track_album_id', 'track_album_name', 
                  'playlist_name', 'playlist_id', 'playlist_genre', 'playlist_subgenre']

    label_encoder = LabelEncoder()

    for c in cols_to_encode:
        ds[c + '_encoded'] = label_encoder.fit_transform(ds[c])

def minmax(ds):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    cls = ['track_popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'duration_s', 'duration_m']
    ds[cls] = scaler.fit_transform(ds[cls])

def remove_duplicates(ds):
    nds = ds.drop_duplicates(subset=['track_id'])

def preprocess_data():

    print(os.path.exists("data/dataset.csv"))
    ds = pd.read_csv("data/dataset.csv")
    missing_values(ds)
    adding_features(ds)
    encoding(ds)
    minmax(ds)
    remove_duplicates(ds)

    ds.to_csv("data/preprocessed_dataset.csv")
    return ds
