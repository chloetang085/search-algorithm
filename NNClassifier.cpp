#include "NNClassifier.h"
#include <iostream>
#include <cmath>
#include <limits>

// Method to train the classifier (store training data and labels)
void NearestNeighborClassifier::train(const std::vector<std::vector<double>>& X_train, const std::vector<int>& y_train) {
    this->X_train = X_train;
    this->y_train = y_train;
}


int NearestNeighborClassifier::test(const std::vector<double>& X_test) {
    double min_distance = std::numeric_limits<double>::infinity();
    int predicted_class = -1;

    // Loop over each training instance and compute the Euclidean distance
    for (size_t i = 0; i < X_train.size(); ++i) {
        double distance = euclidean_distance(X_test, X_train[i]);
        // Suppress this detailed distance output:
        // std::cout << "Distance to instance " << i << ": " << distance 
        //           << " (Class = " << y_train[i] << ")" << std::endl;
        if (distance < min_distance) {
            min_distance = distance;
            predicted_class = y_train[i];
        }
    }

    // Return the class of the nearest neighbor
    return predicted_class;
}



// Method to calculate Euclidean distance between two vectors
double NearestNeighborClassifier::euclidean_distance(const std::vector<double>& v1, const std::vector<double>& v2) {
    double sum = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += std::pow(v1[i] - v2[i], 2);
    }
    return std::sqrt(sum);
}