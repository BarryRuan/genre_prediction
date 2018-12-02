import pandas as pd
import scipy
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from features import get_features

DATASET = 'mxm_dataset.db'
train_x, train_y, test_x, test_y, _, _, _, _ = get_features(DATASET, 'binary', max_songs=120000)
# stack the list of sparse matrices into one big matrix
train_matrix = scipy.sparse.vstack(train_x)
test_matrix = scipy.sparse.vstack(test_x)

k_values = [1, 5, 10, 50, 100, 500, 1000, 5000]
for k in k_values:
	print("K = " + str(k))
	start_time = time.time()
	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(train_matrix, train_y)
	end_time = time.time()
	print("Training time (seconds) : " + str(end_time-start_time))
	preds = neigh.predict(test_matrix)
	print("Accuracy of classifier: " + str(accuracy_score(test_y, preds)))
	print("Precision of classifier: " + str(precision_score(test_y, preds, average='macro')))
	print("Recall of classifier: " + str(recall_score(test_y, preds, average='macro')))
	print("F1-score of classifier: " + str(f1_score(test_y, preds, average='macro')))
	print()

