import os.path
import pandas as pd
import warnings
import numpy as np
from sklearn import metrics
import pickle
from matplotlib import pyplot as plt

from numpy import array
from sklearn.model_selection import KFold, train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import ShuffleSplit

from pipeline import preprocessing
from pipeline.preprocessing import preprocess_data

import warnings

warnings.simplefilter('ignore')


def classification_report(models, X_train, X_test, y_train, y_test):
    for name, model in models.items():
        print(name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))
        print('\n')


def select_model(X_train, X_test, y_train, y_test):
    models = {
        'SVC': SVC(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'RandomForestClassifier': RandomForestClassifier()
    }

    classification_report(models, X_train, X_test, y_train, y_test)


def validate_model(X_train, X_test, y_train, y_test, model):
    y_pred = model.predict(X_test)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()

    dt_classifier = AdaBoostClassifier()
    dt_classifier.fit(X_train, y_train)

    y_pred = dt_classifier.predict(X_test)
    print('Accuracy score: ', metrics.accuracy_score(y_test, y_pred))
    print('Presicion score: ', metrics.precision_score(y_test, y_pred, average=None))
    print('Recall score: ', metrics.recall_score(y_test, y_pred, average=None))
    print('F1 score: ', metrics.f1_score(y_test, y_pred, average=None))
    print(metrics.classification_report(y_test, y_pred))


def save_model(model, name):
    model_pkl_file = f"model/{name}.pkl"
    with open(model_pkl_file, "wb") as file:
        pickle.dump(model, file)


def load_model(name):
    model_pkl_file = f"model/{name}.pkl"
    with open(model_pkl_file, "rb") as file:
        model = pickle.load(file)
        return model


def train_model():
    # ds = preprocessing.preprocess_data()
    ds = pd.read_csv('data/train.csv')
    # model = AdaBoostClassifier(learning_rate=0.05, n_estimators=50, random_state=1717)
    # model = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1)
    model = load_model("GradientBoostingClassifier")
    train(model, ds)
    #cross_validation(ds, model, "GradientBoostingClassifier")

def train(model, ds):
    y_column = 'playlist_genre_encoded'
    X_columns = ['track_popularity', 'danceability', 'energy', 'key',
                 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                 'liveness', 'valence', 'tempo', 'duration_ms',
                 'track_artist_encoded', 'track_album_name_encoded', 'year']
    X = ds[X_columns]
    y = ds[y_column]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.7)

    model.fit(X_train, Y_train)

    save_model(model, "GradientBoostingClassifier")


def feature_importance(models, ds):
    y_column = 'playlist_genre_encoded'
    X_columns = ['track_popularity', 'danceability', 'energy', 'key',
                 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                 'liveness', 'valence', 'tempo', 'duration_ms',
                 'track_artist_encoded', 'track_album_name_encoded', 'year']
    X = ds[X_columns]
    y = ds[y_column]

    for name, model in models.items():
        importances = model.feature_importances_
        indices = np.argsort(importances)
        print("Feature ranking")

        plt.figure(figsize=(5, 10))
        plt.title(f'Feature Importances {name}')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [X_columns[i] for i in indices])
        plt.xlabel('Relative Importance')


def cross_validation(ds, model, name):
    y_column = 'playlist_genre_encoded'
    X_columns = ['track_popularity', 'danceability', 'energy', 'key',
                 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                 'liveness', 'valence', 'tempo', 'duration_ms',
                 'track_artist_encoded', 'track_album_name_encoded', 'year']
    X = ds[X_columns]
    y = ds[y_column]

    ss = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1717)

    for train_index, test_index in ss.split(X):
        classifier = model
        classifier.fit(X.iloc[train_index], y.iloc[train_index])
        y_pred = classifier.predict(X.iloc[test_index])
        print('k-fold set metrics: ', metrics.classification_report(y.iloc[test_index], y_pred))

    save_model(classifier, name)
