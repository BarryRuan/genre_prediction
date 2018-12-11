from data.DataLoader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
  dl = DataLoader()
  train_x, train_y, test_x, test_y = dl.load('binary')

  genres = np.concatenate((train_y, test_y))
  print(type(genres))
  print(genres.shape)

  # read genrenames
  names = []
  datadir = os.path.join(os.path.dirname(__file__), 'data/genresList.txt')
  with open(datadir) as file:
    for line in file:
      names.append(line.strip().lower())
  
  genres = list(map(lambda x: names[x], genres))

  counts = []
  for i in range(len(names)):
    c = len(list(filter (lambda x: x == names[i], genres)))
    counts.append(c)

  plt.bar(names, counts)
  plt.show()
