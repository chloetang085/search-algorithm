#ifndef NNCLASSIFIER_H
#define NNCLASSIFIER_H
using namespace std;
#include <vector>

class NearestNeighborClassifier {
public:
    // Method to train the classifier (store training data and labels)
    void train(const vector<vector<double>>& X_train, const vector<int>& y_train);

    // Method to test the classifier (predict the class of a test instance)
    int test(const vector<double>& X_test);

private:
    // Store the training data and labels
    vector<vector<double>> X_train;
    vector<int> y_train;

    // Method to calculate Euclidean distance between two vectors
    double euclidean_distance(const vector<double>& v1, const vector<double>& v2);
};

#endif 