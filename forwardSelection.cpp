#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip> // for setting decimal precision (googled)

using namespace std;

// function that returns a random accuracy
double evaluate(const vector<int>& featureSet) {
    return 25.0 + static_cast<double>(rand()) / RAND_MAX * 60.0; // random accuracy
}

void forwardSelection(int totalFeatures) {
    cout << "Forward Selection Algorithm\n";
    cout << "Total number of features: " << totalFeatures << "\n";

    vector<int> bestFeatures;
    double bestAccuracy = 0.0;

    for (int step = 0; step < totalFeatures; ++step) {
        int currentBestFeature = -1;
        double currentBestAccuracy = bestAccuracy;

        for (int feature = 1; feature <= totalFeatures; ++feature) {
            if (find(bestFeatures.begin(), bestFeatures.end(), feature) == bestFeatures.end()) {
                // test adding this feature
                vector<int> tempFeatures = bestFeatures;
                tempFeatures.push_back(feature);
                double accuracy = evaluate(tempFeatures);

                cout << "Using feature(s) { ";
                for (int f : tempFeatures) cout << f << " ";
                cout << "} accuracy is " << fixed << setprecision(1) << accuracy << "%\n";

                if (accuracy > currentBestAccuracy) {
                    currentBestFeature = feature;
                    currentBestAccuracy = accuracy;
                }
            }
        }

        if (currentBestFeature != -1) {
            bestFeatures.push_back(currentBestFeature);
            bestAccuracy = currentBestAccuracy;
            cout << "Feature set { ";
            for (int f : bestFeatures) cout << f << " ";
            cout << "} was best, accuracy is " << fixed << setprecision(1) << bestAccuracy << "%\n";
        } else {
            cout << "No improvement in accuracy, stopping search.\n";
            break;
        }
    }

    cout << "Finished search! The best feature subset is { ";
    for (int f : bestFeatures) cout << f << " ";
    cout << "}, which has an accuracy of " << fixed << setprecision(1) << bestAccuracy << "%\n\n";
}

int main() {
    srand(time(0)); // seed random number generator

    int totalFeatures = 4; // example 
    forwardSelection(totalFeatures);

    return 0;
}
