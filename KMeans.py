from sklearn.cluster import KMeans, MiniBatchKMeans
from data.features import LyricsDataSet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import scipy

def run(vectorization_type):
    
    features = LyricsDataSet(vectorization_type)
    train_x = features.get_train_x()
    train_y = features.get_train_y()
    test_x = features.get_test_x()
    test_y = features.get_test_y()
    train_matrix = scipy.sparse.vstack(train_x)
    test_matrix = scipy.sparse.vstack(test_x)
    print(train_matrix.shape[0], test_matrix.shape[0])
    print(type(train_matrix))
    print(type(train_x))

    # K-means algorithm
    km = KMeans(n_clusters=12, random_state=0)
    km = km.fit(train_matrix)
    preds = km.predict(train_matrix)

    aux = []
    curr = []
    ans = []
    for i in range(12):
        for c in range(len(preds)):
            if i == preds[c]:
                curr.append(train_y[c])
        
        max_val = 0
        max_count = -1
        for c in curr:
            if curr.count(c) > max_count:
                max_val = c
                max_count = curr.count(c)

        aux.append(max_val)

    preds = km.predict(test_matrix)
    for i in range(len(preds)):
        ans.append(aux[preds[i]])

    
    # Print stats
    print("Accuracy of classifier: " + str(accuracy_score(test_y, ans)))
    print("Precision of classifier: " + str(precision_score(test_y, ans, average='macro')))
    print("Recall of classifier: " + str(recall_score(test_y, ans, average='macro')))
    print("F1-score of classifier: " + str(f1_score(test_y, ans, average='macro')))
    print()