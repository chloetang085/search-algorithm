#ifndef VALIDATOR_H
#define VALIDATOR_H

#include "NNClassifier.h"
#include <vector>


class Validator {
public:
    // Constructor to initialize the validator with classifier, data, and labels
    Validator(NearestNeighborClassifier& classifier, 
                         const vector<vector<double>>& X, 
                         const vector<int>& y);

    // Method to perform leave-one-out cross-validation and return accuracy
    double validate();

private:
    NearestNeighborClassifier& classifier;
    const vector<vector<double>>& X;
    const vector<int>& y;
};

#endif 