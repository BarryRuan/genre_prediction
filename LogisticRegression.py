import pandas as pd
import scipy
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from features import get_features

DATASET = 'mxm_dataset.db'
train_x, train_y, test_x, test_y, _, _, _, _ = get_features(DATASET, 'binary', max_songs=120000)
# stack the list of sparse matrices into one big matrix
train_matrix = scipy.sparse.vstack(train_x)
test_matrix = scipy.sparse.vstack(test_x)

c_values = [.0001, .001, .01, .1, 1, 10, 100]
for c in c_values:
	print("C = " + str(c))
	start_time = time.time()
	lr = LogisticRegression(C=c)
	lr.fit(train_matrix, train_y)
	end_time = time.time()
	print("Training time (seconds) : " + str(end_time-start_time))
	preds = lr.predict(test_matrix)
	print("Accuracy of classifier: " + str(accuracy_score(test_y, preds)))
	print("Precision of classifier: " + str(precision_score(test_y, preds, average='macro')))
	print("Recall of classifier: " + str(recall_score(test_y, preds, average='macro')))
	print("F1-score of classifier: " + str(f1_score(test_y, preds, average='macro')))
	print()
