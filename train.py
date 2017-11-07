import utils

if __name__ == '__main__':

    dir = "log/peps normal"
    pattern = r'(left|right|front|back|start|trunk|lock)\\\d{1,2}.csv$'
    pattern_valid = r'(3|6|9|12).csv$'
    utils.construct_set(dir, pattern, pattern_valid, filter=1)
    utils.save_to_csv()

    X, y = utils.load_all()
    X_train, X_valid, y_train, y_valid = utils.load_train_valid()

    # compare train result
    methods = ["Logistic", "LDA", "QDA", "KNN", "SVM", "RF", "GBM", "MLP"]
    params = [None,
              None,
              None,
              {"n_neighbors": 10},
              {"C": 0.25, "gamma": 0.5},
              {"max_features": 2, "n_estimators": 100},
              {"n_estimators": 400, "max_depth": 3},
              {"hidden_layer_sizes": (16, 8)}]
    for method, param in zip(methods, params):
        utils.train(X_train, X_valid, y_train, y_valid, method=method, param=param)
