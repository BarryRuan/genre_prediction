#Grene Prediction

-----------------------------------------
##USAGE:

To run the prediction pipeline with specific feature representation and method

```
python3 main.py [binary | frequency | tf-idf] [naive_bayes | logistic_regression | knn | svm | baseline | random_forest | gradient_boosted_tree | neural_network | k_means]
```


-----------------------------------------
DEVELOPMENT: 

To get feature maps, use the following codes:

  (1). for binary features:

      ```ruby
      from data.features import LyricsDataSet
      lyricsData = LyricsDataSet('binary')
      ```

  (2). for term frequency: 

      ```ruby
      from data.features import LyricsDataSet
      lyricsData = LyricsDataSet('frequency')
      ```

  (3). for tf-idf:

      ```ruby
      from data.features import LyricsDataSet
      lyricsData = LyricsDataSet('tf-idf')
      ```

  To get train and test splits, use the following codes:

    ```ruby
    train_x = lyricsData.get_train_x()
    train_y = lyricsData.get_train_y()
    test_x = lyricsData.get_test_x()
    test_y = lyricsData.get_test_y()
    ```

  Note that train_x and test_x are lists of scipy.sparse_matrix, to 
  convert it to np.array(), you can use ```train_x[i].toarray()```

------------------
|db_processing.py|
------------------
This file is used to transform the original dataset mxm_dataset.db to
several .txt files including data/frequency_features.txt, 
data/vocabulary.txt, data/genreList.txt. The new files are much smaller
to the original file (<60MB vs. 2.6GB) so that time consumed on reading
train and test split could be greatly reduced from (10 min to 20s).
If you downloaded the data/ directory, there is no need to run this code.
