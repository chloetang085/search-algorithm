#ifndef FEATURESELECTION_H
#define FEATURESELECTION_H

#include <vector>

void forwardSelection(int totalFeatures, const std::vector<std::vector<double>>& X, const std::vector<int>& y);
void backwardElimination(int totalFeatures, const std::vector<std::vector<double>>& X, const std::vector<int>& y);

#endif // FEATURESELECTION_H