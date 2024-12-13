#ifndef FEATURESELECTION_H
#define FEATURESELECTION_H

#include <vector>

using namespace std;

void forwardSelection(int totalFeatures, const std::vector<vector<double>>& X, const vector<int>& y);
void backwardElimination(int totalFeatures, const std::vector<vector<double>>& X, const vector<int>& y);

#endif // FEATURESELECTION_H
