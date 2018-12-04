import numpy as np
from scipy import sparse
import scipy
import sys
import sqlite3

class LyricsDataSet:
    def __init__(self, feature_type='binary',max_song=None):
        self.train_x, self.train_y, self.test_x, self.test_y = \
                self._get_features(feature_type, max_song)
        len_train = int(len(self.train_x)*0.8)
        self.dev_x = self.train_x[len_train:]
        self.dev_y = self.train_y[len_train:]
        self.train_x = self.train_x[:len_train]
        self.train_y = self.train_y[:len_train]
        self.train_count = 0
        self.dev_count = 0
        self.test_count = 0


    def _read_idf(self):
        f = open('data/vocabulary.txt')
        idf = []
        for line in f.readlines():
            idf.append(float(line.split(',')[1]))
        f.close()
        return np.array(idf)

    def _get_features(self, feature_type, max_song):
        print("Getting train and test split. (features:{})"\
                .format(feature_type))
        f = open('data/frequency_features.txt')
        lines = f.readlines()
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        if max_song:
            lines = lines[:max_song]
        if feature_type == 'tf-idf':
            idf = self._read_idf()
        num_songs = 0
        for line in lines:
            if num_songs != 0 and num_songs % 10000 == 0:
                print("{} songs read.".format(num_songs))
            num_songs += 1
            genre, is_test, wc_pairs = line.split('+')
            wc_pairs = wc_pairs.split(' ')
            words = []
            counts = []
            for wc_pair in wc_pairs:
                word, count = wc_pair.split(',')
                words.append(int(word))
                if feature_type == 'binary':
                    counts.append(1)
                else:
                    counts.append(int(count))
            if feature_type == 'tf-idf':
                counts = (np.array(counts) * idf[words]).tolist()
            z = np.zeros(len(words))
            feature_vector = sparse.coo_matrix((counts,(z,words)),shape=(1,5000))
            if is_test == '1':
                test_x.append(feature_vector)
                test_y.append(int(genre))
            else:
                train_x.append(feature_vector)
                train_y.append(int(genre))
        f.close()
        return np.array(train_x),\
                np.array(train_y),\
                np.array(test_x),\
                np.array(test_y)

    def get_train_x(self):
        return self.train_x

    def get_train_y(self):
        return self.train_y

    def get_test_x(self):
        return self.test_x

    def get_test_y(self):
        return self.test_y

    def get_dev_x(self):
        return self.dev_x

    def get_dev_y(self):
        return self.dev_y

    def get_batch(self, partition='train', batch_size=128):
        if partition == 'train':
            batchx, batchy, self.train_x, self.train_y, self.train_count = \
                self._batch_helper(
                    self.train_x, self.train_y, self.train_count, batch_size)
            return batchx, batchy
        elif partition == 'dev':
            batchx, batchy, self.dev_x, self.dev_y, self.dev_count = \
                self._batch_helper(
                    self.dev_x, self.dev_y, self.dev_count, batch_size)
            return batchx, batchy
        elif partition == 'test':
            batchX, self.test_count = \
                self._batch_helper(
                    self.testX, None, self.test_count, batch_size)
            return batchX

    def _batch_helper(self, X, y, count, batch_size):
        if count + batch_size > len(X):
            if type(y) == np.ndarray:
                count = 0
                rand_idx = np.random.permutation(len(X))
                X = X[rand_idx]
                y = y[rand_idx]
        batchX = X[count:count+batch_size]
        if type(y) == np.ndarray:
            batchY = y[count:count+batch_size]
        count += batch_size
        if type(y) == np.ndarray:
            return scipy.sparse.vstack(batchX).toarray(),\
                    np.eye(15)[batchY], X, y, count
        else:
            return scipy.sparse.vstack(batchX).toarray(), count

    def test_done(self):
        result = self.test_count >= len(self.test_x)
        if result:
            self.test_count = 0
        return result
