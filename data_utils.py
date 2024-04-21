import pandas as pd
import numpy as np


def data_to_matrices(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    unique_users = pd.concat([train['userId'], test['userId']]).unique()
    unique_movies = pd.concat([train['movieId'], test['movieId']]).unique()
    n, d = len(unique_users), len(unique_movies)
    Z = np.full((n, d), np.nan)
    T = np.full((n, d), np.nan)

    user_ids = {user: index for index, user in enumerate(sorted(unique_users))}
    movie_ids = {movie: index for index, movie in enumerate(sorted(unique_movies))}

    for _, row in train.iterrows():
        user_index = user_ids[row['userId']]
        movie_index = movie_ids[row['movieId']]
        Z[user_index, movie_index] = row['rating']

    for _, row in test.iterrows():
        user_index = user_ids[row['userId']]
        movie_index = movie_ids[row['movieId']]
        T[user_index, movie_index] = row['rating']

    return Z, T


def Z_prime(Z, T):
    Zprime = np.copy(Z)
    mask = np.isnan(T)
    Zprime[mask] = np.nan
    return Zprime


def nan_to_user_movie_mean(matrix):
    user_means = np.nanmean(matrix, axis=1, keepdims=True)
    user_means = np.nan_to_num(user_means, nan=0)
    movie_means = np.nanmean(matrix, axis=0, keepdims=True)
    all_nan_columns = np.isnan(movie_means)
    movie_means = np.where(all_nan_columns, user_means, movie_means)
    combined_means = (4 * user_means + movie_means) / 5
    nan_replaced = np.where(np.isnan(matrix), combined_means, matrix)
    return nan_replaced
