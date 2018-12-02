import numpy as np
import scipy

def get_features(dataset, feature_type='binary'):
    train_x = np.zeros(100,100)
    train_y = np.zeros(100,100)
    test_x = np.zeros(100,100)
    test_y = np.zeros(100,100)
    return train_x, train_y, test_x, test_y

def get_vocabulary(dataset):
    vocabulary = {}
    return vocabulary
