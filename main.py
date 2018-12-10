# Written by Preet Patel (ppreet), Zhiming Ruan (ruanzhim), Lance Strait (straitl) and Nathan Wernert (wernertn) for EECS 595 Final Project

import sys
import KNN, LogisticRegression, NaiveBayes, SVM, Baseline, RandomForest, GradientBoosting, NN, KMeans


# Globals
vectorization_list = ["binary", "frequency", "tf-idf"]
classification_list = ["knn", "logistic_regression", "naive_bayes", "svm", "baseline", "random_forest", "gradient_boosted_tree", "neural_network", "k_means"]

if __name__ == "__main__":

    ###### Check and get command line arguments ######

    if len(sys.argv) != 3:
        print("Error: Check command line input")
        exit(1)
    
    vectorization_in = sys.argv[1]
    classifier_in = sys.argv[2]

    if vectorization_in not in vectorization_list:
        print("Error: Vectorization option not valid")
        exit(1)
    
    if classifier_in not in classification_list:
        print("Error: Classification option not valid")
        exit(1)

    ##### Call the classifier #####

    if classifier_in == "knn":
        KNN.run(vectorization_in)
    
    elif classifier_in == "logistic_regression":
        LogisticRegression.run(vectorization_in)

    elif classifier_in == "naive_bayes":
        NaiveBayes.run(vectorization_in)

    elif classifier_in == "svm":
        SVM.run(vectorization_in)

    elif classifier_in == "baseline":
        rf = Baseline.Baseline(vectorization_in)
        rf.run()

    elif classifier_in == "random_forest":
        rf = RandomForest.RandomForest(vectorization_in)
        args = {
            'n_estimators': 100
        }
        rf.run(args)
    
    elif classifier_in == "gradient_boosted_tree":
        rf = GradientBoosting.GradientBoosting(vectorization_in)
        rf.run()
    
    elif classifier_in == "neural_network":
        NN.run(vectorization_in)
    
    elif classifier_in == "k_means":
        KMeans.run(vectorization_in)


    


