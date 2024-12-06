#include "Validator.h"
#include <iostream>
#include <chrono> // Include for measuring time (googled)

Validator::Validator(NearestNeighborClassifier& classifier, 
                                           const vector<vector<double>>& X, 
                                           const vector<int>& y)
    : classifier(classifier), X(X), y(y) {}

double Validator::validate() {
    size_t correct_predictions = 0;
    size_t n = X.size();

    cout << "Instance | Predicted | Actual | Correct | Time Elapsed (ms)" << endl;

    // Loop over each instance in the dataset
    for (size_t i = 0; i < n; ++i) {
        // Start the timer for this prediction
        auto start_time = chrono::high_resolution_clock::now();

        // Create training and test data
        vector<vector<double>> X_train = X;
        vector<int> y_train = y;
        vector<double> X_test = X[i];
        int y_test = y[i];

        // Remove the i-th instance from training data
        X_train.erase(X_train.begin() + i);
        y_train.erase(y_train.begin() + i);

        // Train the classifier on the remaining data
        classifier.train(X_train, y_train);

        // Test the classifier on the left-out instance
        int predicted = classifier.test(X_test);

        // End the timer for this prediction
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed_time = end_time - start_time;

        // Check if the prediction was correct
        bool is_correct = (predicted == y_test);
        if (is_correct) {
            ++correct_predictions;
        }

        // Print trace for this instance with elapsed time
        cout << i << "        | " << predicted << "         | " << y_test 
                  << "      | " << (is_correct ? "Yes" : "No") 
                  << "      | " << elapsed_time.count() << " ms" << endl;
    }

    // Return the accuracy as a fraction of correct predictions
    return static_cast<double>(correct_predictions) / n;
}
