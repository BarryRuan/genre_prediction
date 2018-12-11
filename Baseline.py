from data.DataLoader import DataLoader
import numpy as np
import sklearn.metrics
import scipy.sparse
import sys
from collections import Counter

# Tune max depth -- decrease == robust?
# for GBT -> tune max features
class Baseline:
  def __init__(self, method):
    print('USING METHOD: {}'.format(method))
    # Read lyrics dataset and get train/test splits
    dl = DataLoader()
    self.train_x, self.train_y, self.test_x, self.test_y = dl.load(method)

  def train(self, args):
    print('\nTraining BaselineClassifier')

    counts = [Counter() for i in range(self.train_x.shape[1])]
    nonzero = self.train_x.nonzero()
    for i in range(len(nonzero[0])):
      sid = nonzero[0][i]
      wid = nonzero[1][i]
      gid = self.train_y[sid]
      counts[wid][gid] += 1

    genre = [0 for i in range(self.train_x.shape[1])]
    for i in range(len(genre)):
      max_count = None
      max_genre = None
      for gid, count in dict(counts[i]).items():
        if max_count == None or count > max_count:
          max_count = count
          max_genre = gid
      genre[i] = max_genre
    self.genre = genre

  def predict(self):
    print('\nGetting Predictions')

    counts = [Counter() for i in range(self.test_x.shape[0])]
    nonzero = self.test_x.nonzero()
    for i in range(len(nonzero[0])):
      sid = nonzero[0][i]
      wid = nonzero[1][i]
      pred = self.genre[wid]
      counts[sid][pred] += 1

    self.predictions = [None for i in range(self.test_x.shape[0])]
    for i in range(len(self.predictions)):
      max_count = None
      max_genre = None
      for gid, count in dict(counts[i]).items():
        if max_count == None or count > max_count:
          max_count = count
          max_genre = gid
      self.predictions[i] = max_genre

    print(set(self.predictions))


  def metrics(self):
    a = sklearn.metrics.accuracy_score(self.test_y, self.predictions)
    p = sklearn.metrics.precision_score(self.test_y, self.predictions, average='weighted')
    r = sklearn.metrics.recall_score(self.test_y, self.predictions, average='weighted')
    f = sklearn.metrics.f1_score(self.test_y, self.predictions, average='weighted')
    return a, p, r, f

  def run(self, args={}):
    self.train(args)
    self.predict()
    a, p, r, f = self.metrics()
    print('\naccuracy: {}'.format(a))
    print('precision: {}'.format(p))
    print('recall: {}'.format(r))
    print('f1: {}'.format(f))
