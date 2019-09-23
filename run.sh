## vectorizations and classifiers
declare -a vecs=("binary" "frequency" "tf-idf")
declare -a class=("knn" "logistic_regression" "naive_bayes" "svm")

## Loop
for v in "${vecs[@]}"
do
    for c in "${class[@]}"
        do
            echo "Running $v $c"
            python main.py "$v" "$c"
            echo
        done
done