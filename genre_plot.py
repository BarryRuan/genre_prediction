from data.DataLoader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
  dl = DataLoader()
  train_x, train_y, test_x, test_y = dl.load('binary')

  genres = np.concatenate((train_y, test_y))
  print(len(genres))

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
  
  plt.rc('axes', axisbelow=True)
  plt.grid(b=True, axis='x', color='#eeeeee', zorder=-1)
  plt.ylabel('Genre', labelpad=15, fontsize=18, color='#555555')
  plt.xlabel('Number of songs', labelpad=15, fontsize=18, color='#555555')
  plt.title('Genre Distribution', pad=15, fontsize=20, color='#555555')
  plt.xticks(fontsize=14, color='#555555')
  plt.yticks(fontsize=14, color='#555555')
  plt.barh(names, counts)
  plt.show()
