#include "Validator.h"
#include <iostream>
#include <cmath>

Validator::Validator(NearestNeighborClassifier& classifier, 
                     const std::vector<std::vector<double>>& X, 
                     const std::vector<int>& y,
                     const std::vector<int>& featureSubset)
    : classifier(classifier), X(X), y(y), featureSubset(featureSubset) {}

double Validator::validate() {
    size_t correct_predictions = 0;
    size_t n = X.size();

    for (size_t i = 0; i < n; ++i) {
        std::vector<std::vector<double>> X_train = X;
        std::vector<int> y_train = y;
        std::vector<double> X_test;

        X_train.erase(X_train.begin() + i);
        y_train.erase(y_train.begin() + i);

        // Filter features for test instance
        for (int idx : featureSubset) {
            X_test.push_back(X[i][idx - 1]); // Convert 1-based to 0-based
        }

        // Filter features for training data
        std::vector<std::vector<double>> filtered_X_train;
        for (const auto& row : X_train) {
            std::vector<double> filtered_row;
            for (int idx : featureSubset) {
                filtered_row.push_back(row[idx - 1]); // Convert 1-based to 0-based
            }
            filtered_X_train.push_back(filtered_row);
        }

        classifier.train(filtered_X_train, y_train);
        int predicted = classifier.test(X_test);

        if (predicted == y[i]) {
            ++correct_predictions;
        }
    }

    return static_cast<double>(correct_predictions) / n;
}