# encoding=utf8
import os
import re
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

class Data(object):
    def __init__(self, dict_path=None, train_data='data/your_train_data.pkl'):
        self.max_len = 200 
        with open(dict_path, 'rb') as inp:
            self.word2id = pickle.load(inp)
            self.id2word = pickle.load(inp)
            self.tag2id = pickle.load(inp)
            self.id2tag = pickle.load(inp)

        with open(train_data, 'rb') as inp:
            self.X = pickle.load(inp)
            self.Y = pickle.load(inp)

    def builderTrainData(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        print(
            'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
                X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

        print('Creating the data generator ...')
        data_train = BatchGenerator(X_train, y_train, shuffle=True)
        data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
        data_test = BatchGenerator(X_test, y_test, shuffle=False)
        print('Finished creating the data generator.')

        return data_train, data_valid, data_test

class BatchGenerator(object):

    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]