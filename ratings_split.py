import pandas as pd
import argparse
import random


def split_ratings(input_file, train_output, test_output):
    ratings = pd.read_csv(input_file)
    train_file = pd.DataFrame()
    test_file = pd.DataFrame()

    for ID in ratings['userId'].unique():
        user_ratings = ratings[ratings['userId'] == ID]
        length = len(user_ratings)
        train_indexes = random.sample(range(length), round(0.9 * length))
        train_indexes.sort()
        train_file = pd.concat([train_file, user_ratings.iloc[train_indexes]])
        test_file = pd.concat([test_file, user_ratings.drop(user_ratings.index[train_indexes])])

    train_file.to_csv(train_output, index=False)
    test_file.to_csv(test_output, index=False)
    print(f"Training set saved to {train_output}, test set saved to {test_output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a ratings CSV into training and testing sets.')
    parser.add_argument('input_file', type=str, help='The CSV file to be split.')
    parser.add_argument('train_output', type=str, help='The CSV file to store the training set.')
    parser.add_argument('test_output', type=str, help='The CSV file to store the testing set.')

    args = parser.parse_args()
    split_ratings(args.input_file, args.train_output, args.test_output)
