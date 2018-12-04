import os
from data.features import LyricsDataSet
import scipy.sparse
import numpy as np

class DataLoader:
  def __init__(self):
    self.methods = ['binary', 'frequency', 'tf-idf']
    self.dirname = os.path.join(os.path.dirname(__file__), 'matrices')

    if not os.path.exists(self.dirname):
        os.makedirs(self.dirname)
        self.create()

    
  def load(self, method):
    """Returns train_x, train_y, test_x, and test_y for a given method"""
    valid = ['binary', 'frequency', 'tf-idf']
    if method not in valid:
        raise ValueError(
            '{} is not an accepted method. Valid methods are {}'.format(method, valid))
    
    train_x = scipy.sparse.load_npz(os.path.join(self.dirname, '{}-train_x.npz'.format(method)))
    train_y = np.load(os.path.join(self.dirname, '{}-train_y.npy'.format(method)))
    test_x = scipy.sparse.load_npz(os.path.join(self.dirname, '{}-test_x.npz'.format(method)))
    test_y = np.load(os.path.join(self.dirname, '{}-test_y.npy'.format(method)))
    return train_x, train_y, test_x, test_y

  def create(self):
    print('Saving matrices to: {}'.format(self.dirname))
    # for each method, save scipy sparse matrices for x, numpy arrays for y
    for method in self.methods:
      # Read lyrics dataset and get train/test splits
      data = LyricsDataSet(method)
      train_x = data.get_train_x()
      train_y = data.get_train_y()
      test_x = data.get_test_x()
      test_y = data.get_test_y()

      # convert x-lists into sparse matrices
      train_x = scipy.sparse.bmat([[x] for x in train_x])
      scipy.sparse.save_npz(
          'data/matrices/{}-train_x'.format(method), train_x)
      np.save('data/matrices/{}-train_y'.format(method), np.array(train_y))
      test_x = scipy.sparse.bmat([[x] for x in test_x])
      scipy.sparse.save_npz(
          'data/matrices/{}-test_x'.format(method), test_x)
      np.save('data/matrices/{}-test_y'.format(method), np.array(test_y))
