from data.DataLoader import DataLoader
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics
import scipy.sparse
# Tune max depth -- decrease == robust?
# for GBT -> tune max features
class GradientBoosting:
  def __init__(self, method):
    # Read lyrics dataset and get train/test splits
    dl = DataLoader()
    self.train_x, self.train_y, self.test_x, self.test_y = dl.load(method)

  def train(self, args):
    print('\nTraining GradientBoostingClassifier')
    self.clf = GradientBoostingClassifier(**args)
    self.clf.fit(self.train_x, self.train_y)

  def predict(self):
    print('\nGetting Predictions')
    self.predictions = self.clf.predict(self.test_x)

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

if __name__ == '__main__':
  rf = GradientBoosting('binary')
  rf.run()