#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
using namespace std;

// Declare the evaluate function as external (defined in main.cpp)
extern double evaluate(const vector<int>& featureSet);

void backwardElimination(int totalFeatures) {
    cout << "\nRunning Backward Elimination...\n";

    // Initialize the full set of features
    vector<int> features;
    for (int i = 1; i <= totalFeatures; ++i) features.push_back(i);

    // Calculate accuracy with all features
    double bestAccuracy = evaluate(features);
    cout << "Using all features { ";
    for (int f : features) cout << f << " ";
    cout << "}, initial accuracy is " << fixed << setprecision(1) << bestAccuracy << "%\n";
    cout << "\nBeginning search.\n";

    while (features.size() > 1) {
        int worstFeatureToRemove = -1;
        double currentBestAccuracy = bestAccuracy;

        cout << "Testing removal of each feature:\n";

        for (int feature : features) {
            // Create a temporary feature set without the current feature
            vector<int> tempFeatures = features;
            tempFeatures.erase(remove(tempFeatures.begin(), tempFeatures.end(), feature), tempFeatures.end());
            double accuracy = evaluate(tempFeatures);

            // Trace the results of removing each feature
            cout << "  Removing feature " << feature << " -> accuracy with feature(s) { ";
            for (int f : tempFeatures) cout << f << " ";
            cout << "} is " << fixed << setprecision(1) << accuracy << "%\n";

            // Keep track of the feature whose removal improves accuracy the most
            if (accuracy > currentBestAccuracy) {
                worstFeatureToRemove = feature;
                currentBestAccuracy = accuracy;
            }
        }

        if (worstFeatureToRemove != -1) {
            // Remove the worst feature and update the best accuracy
            features.erase(remove(features.begin(), features.end(), worstFeatureToRemove), features.end());
            bestAccuracy = currentBestAccuracy;
            cout << "\nFeature set { ";
            for (int f : features) cout << f << " ";
            cout << "} was best, accuracy is " << fixed << setprecision(1) << bestAccuracy << "%\n";
        } else {
            cout << "(No improvement, stopping search.)\n";
            break;
        }
    }

    // Final output
    cout << "\nFinished search! The best feature subset is { ";
    for (int f : features) cout << f << " ";
    cout << "}, which has an accuracy of " << fixed << setprecision(1) << bestAccuracy << "%\n";
}