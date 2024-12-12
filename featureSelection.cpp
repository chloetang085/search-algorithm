#include "featureSelection.h"
#include "Validator.h"
#include "NNClassifier.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>

using namespace std;

double evaluateFeatureSet(const vector<int>& featureSet, const vector<vector<double>>& X, const vector<int>& y) {
    // Initialize NNClassifier and Validator
    NearestNeighborClassifier nn_classifier;
    Validator validator(nn_classifier, X, y, featureSet);
    return validator.validate();
}

void forwardSelection(int totalFeatures, const vector<vector<double>>& X, const vector<int>& y) {
    cout << "\nRunning Forward Selection...\n";

    vector<int> currentFeatures;
    double bestAccuracy = 0.0;

    for (int i = 1; i <= totalFeatures; ++i) {
        int bestFeature = -1;
        double maxAccuracy = bestAccuracy;

        for (int f = 1; f <= totalFeatures; ++f) {
            if (find(currentFeatures.begin(), currentFeatures.end(), f) != currentFeatures.end()) {
                continue; // Skip already selected features
            }

            vector<int> tempFeatures = currentFeatures;
            tempFeatures.push_back(f);

            double accuracy = evaluateFeatureSet(tempFeatures, X, y);
            cout << "Using feature(s) { ";
            for (int feat : tempFeatures) cout << feat << " ";
            cout << "} accuracy is " << fixed << setprecision(2) << accuracy * 100 << "%\n";

            if (accuracy > maxAccuracy) {
                maxAccuracy = accuracy;
                bestFeature = f;
            }
        }

        if (bestFeature != -1) {
            currentFeatures.push_back(bestFeature);
            bestAccuracy = maxAccuracy;
            cout << "Feature set { ";
            for (int feat : currentFeatures) cout << feat << " ";
            cout << "} was best, accuracy is " << fixed << setprecision(2) << bestAccuracy * 100 << "%\n";
        } else {
            break;
        }
    }

    cout << "\nBest Feature Subset (Forward Selection): { ";
    for (int feat : currentFeatures) cout << feat << " ";
    cout << "} | Accuracy: " << fixed << setprecision(2) << bestAccuracy * 100 << "%\n";
}

void backwardElimination(int totalFeatures, const vector<vector<double>>& X, const vector<int>& y) {
    cout << "\nRunning Backward Elimination...\n";

    vector<int> currentFeatures;
    for (int i = 1; i <= totalFeatures; ++i) currentFeatures.push_back(i);

    double bestAccuracy = evaluateFeatureSet(currentFeatures, X, y);
    cout << "Using all features { ";
    for (int f : currentFeatures) cout << f << " ";
    cout << "}, initial accuracy is " << fixed << setprecision(2) << bestAccuracy * 100 << "%\n";

    while (currentFeatures.size() > 1) {
        int worstFeature = -1;
        double maxAccuracy = 0.0;

        for (int f : currentFeatures) {
            vector<int> tempFeatures = currentFeatures;
            tempFeatures.erase(remove(tempFeatures.begin(), tempFeatures.end(), f), tempFeatures.end());

            double accuracy = evaluateFeatureSet(tempFeatures, X, y);
            cout << "Removing feature " << f << " -> accuracy with feature(s) { ";
            for (int feat : tempFeatures) cout << feat << " ";
            cout << "} is " << fixed << setprecision(2) << accuracy * 100 << "%\n";

            if (accuracy > maxAccuracy) {
                maxAccuracy = accuracy;
                worstFeature = f;
            }
        }

        if (worstFeature != -1) {
            currentFeatures.erase(remove(currentFeatures.begin(), currentFeatures.end(), worstFeature), currentFeatures.end());
            bestAccuracy = maxAccuracy;
            cout << "Feature set { ";
            for (int feat : currentFeatures) cout << feat << " ";
            cout << "} was best, accuracy is " << fixed << setprecision(2) << bestAccuracy * 100 << "%\n";
        } else {
            break;
        }
    }

    cout << "\nBest Feature Subset (Backward Elimination): { ";
    for (int feat : currentFeatures) cout << feat << " ";
    cout << "} | Accuracy: " << fixed << setprecision(2) << bestAccuracy * 100 << "%\n";
}