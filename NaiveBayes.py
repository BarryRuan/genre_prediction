from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from data.features import LyricsDataSet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import scipy

features = LyricsDataSet('binary')
train_x = features.get_train_x()
train_y = features.get_train_y()
test_x = features.get_test_x()
test_y = features.get_test_y()
train_matrix = scipy.sparse.vstack(train_x)
test_matrix = scipy.sparse.vstack(test_x)

# Naive Bayes algorithm
naive_bayes = MultinomialNB()
naive_bayes = naive_bayes.fit(train_matrix, train_y)
preds = naive_bayes.predict(test_matrix)

# Print stats
print("Accuracy of classifier: " + str(accuracy_score(test_y, preds)))
print("Precision of classifier: " + str(precision_score(test_y, preds, average='macro')))
print("Recall of classifier: " + str(recall_score(test_y, preds, average='macro')))
print("F1-score of classifier: " + str(f1_score(test_y, preds, average='macro')))
print()