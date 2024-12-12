#ifndef VALIDATOR_H
#define VALIDATOR_H

#include "NNClassifier.h"
#include <vector>

class Validator {
private:
    NearestNeighborClassifier& classifier;
    const std::vector<std::vector<double>>& X;
    const std::vector<int>& y;
    std::vector<int> featureSubset; // Added featureSubset for filtering

public:
    Validator(NearestNeighborClassifier& classifier, 
              const std::vector<std::vector<double>>& X, 
              const std::vector<int>& y,
              const std::vector<int>& featureSubset);

    double validate();
};

#endif