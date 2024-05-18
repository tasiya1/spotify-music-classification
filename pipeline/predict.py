import pandas as pd
import pickle
from pipeline.train_model import load_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def test_model():
    #data = pd.read_csv('data/dataset.csv')
    #ds = preprocessing.preprocess_data()

    y_column = 'playlist_genre_encoded'
    X_columns = ['track_album_id_encoded', 'track_id_encoded', 'track_name_encoded', 'Unnamed: 0', 'track_popularity', 'danceability', 'energy', 'key',
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms',
        'track_artist_encoded', 'track_album_name_encoded', 'year']
    
    ds = pd.read_csv('data/new_input.csv')

    y_column = 'playlist_genre_encoded'
    X_columns = ['track_popularity', 'danceability', 'energy', 'key',
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms',
        'track_artist_encoded', 'track_album_name_encoded', 'year']
    X = ds[X_columns]
    y = ds[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1717)

    #model = load_model("AdaBoostingClassifier")
    model = load_model("GradientBoostingClassifier")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
