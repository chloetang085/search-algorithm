#include <iostream>
#include <vector>
#include <iomanip> //googled
using namespace std;

// Declare the evaluate function as external (defined in main.cpp)
extern double evaluate(const vector<int>& featureSet);

void forwardSelection(int totalFeatures) {
    cout << "\nRunning Forward Selection...\n";
    cout << "Using no features and \"random\" evaluation, I get an accuracy of 55.4%\n";
    cout << "\nBeginning search.\n";

    vector<int> bestFeatures;
    double bestAccuracy = 0.0;

    for (int step = 0; step < totalFeatures; ++step) {
        int currentBestFeature = -1;
        double currentBestAccuracy = bestAccuracy;

        for (int feature = 1; feature <= totalFeatures; ++feature) {
            if (find(bestFeatures.begin(), bestFeatures.end(), feature) == bestFeatures.end()) {
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
            cout << "\nFeature set { ";
            for (int f : bestFeatures) cout << f << " ";
            cout << "} was best, accuracy is " << fixed << setprecision(1) << bestAccuracy << "%\n";
        } else {
            cout << "No improvement in accuracy, stopping search.\n";
            break;
        }
    }

    cout << "\nFinished search! The best feature subset is { ";
    for (int f : bestFeatures) cout << f << " ";
    cout << "}, which has an accuracy of " << fixed << setprecision(1) << bestAccuracy << "%\n";
}