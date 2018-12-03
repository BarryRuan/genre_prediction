import numpy as np
from scipy import sparse
import scipy
import sys
import sqlite3

class LyricsDataSet:
    def __init__(self, feature_types='binary',max_song=None):
        self.train_x, self.train_y, self.test_x, self.test_y = \
                self._get_features(feature_types, max_song)

    def _read_idf():
        f = open('vocabulary.txt')
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
            idf = _read_idf()
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
        return train_x, train_y, test_x, test_y

    def get_train_x(self):
        return self.train_x

    def get_train_y(self):
        return self.train_y

    def get_test_x(self):
        return self.test_x

    def get_test_y(self):
        return self.test_y
