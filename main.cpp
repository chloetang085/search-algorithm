#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <set>
#include <cstdlib>
#include <iomanip>
#include <ctime>
#include "NNClassifier.h"
#include "Validator.h"
#include "FeatureSelection.h"

using namespace std;

// Function to load data from a file
void loadDataset(const string& filename, vector<vector<double>>& X, vector<int>& y) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open the file: " + filename);
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int label;
        ss >> label;
        y.push_back(label);

        vector<double> features;
        double value;
        while (ss >> value) {
            features.push_back(value);
        }
        X.push_back(features);
    }
}

// Function to normalize features (zero mean, unit variance)
void normalize_features(vector<vector<double>>& X) {
    size_t num_features = X[0].size();
    vector<double> mean(num_features, 0.0);
    vector<double> dev(num_features, 0.0);

    // Calculate mean
    for (const auto& row : X) {
        for (size_t i = 0; i < num_features; ++i) {
            mean[i] += row[i];
        }
    }
    for (size_t i = 0; i < num_features; ++i) {
        mean[i] /= X.size();
    }

    // Calculate standard deviation
    for (const auto& row : X) {
        for (size_t i = 0; i < num_features; ++i) {
            dev[i] += pow(row[i] - mean[i], 2);
        }
    }
    for (size_t i = 0; i < num_features; ++i) {
        dev[i] = sqrt(dev[i] / X.size());
        if (dev[i] == 0) {
            dev[i] = 1.0; // Prevent division by zero
        }
    }

    // Normalize each feature
    for (auto& row : X) {
        for (size_t i = 0; i < num_features; ++i) {
            row[i] = (row[i] - mean[i]) / dev[i];
        }
    }
}

// Function to perform randomized search
void randomizedSearch(int totalFeatures, const vector<vector<double>>& X, const vector<int>& y, int maxIterations) {
    srand(time(nullptr)); // Seed for random number generation
    set<int> bestFeatureSubset;
    double bestAccuracy = 0.0;

    for (int i = 0; i < maxIterations; ++i) {
        set<int> randomFeatures;
        int numFeatures = rand() % totalFeatures + 1; // Random number of features

        // Generate a random subset of features
        while (randomFeatures.size() < numFeatures) {
            int feature = rand() % totalFeatures + 1;
            randomFeatures.insert(feature);
        }

        // Evaluate the subset (dummy accuracy for demonstration, replace with actual evaluation)
        double accuracy = static_cast<double>(rand()) / RAND_MAX;

        cout << "Iteration " << i + 1 << ": Feature subset {";
        for (int feature : randomFeatures) cout << feature << " ";
        cout << "} Accuracy: " << accuracy << "\n";

        if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            bestFeatureSubset = randomFeatures;
        }
    }

    cout << "Best feature subset: {";
    for (int feature : bestFeatureSubset) cout << feature << " ";
    cout << "} with accuracy: " << bestAccuracy * 100 << "%\n";
}

int main() {

    vector<vector<double>> X;
    vector<int> y;

    cout << "Select the dataset to use:\n";
    cout << "1. Small Dataset\n";
    cout << "2. Large Dataset\n";
    cout << "3. Titanic Dataset\n";
    cout << "Enter your choice (1, 2, or 3): ";
    int datasetChoice;
    cin >> datasetChoice;

    string filename;
    if (datasetChoice == 1) {
        filename = "small-test-dataset.txt"; // Small dataset
    } else if (datasetChoice == 2) {
        filename = "large-test-dataset.txt"; // Large dataset
    } else if (datasetChoice == 3) {
        filename = "titanic-clean.txt"; // Titanic dataset
    } else {
        cerr << "Invalid choice. Exiting program.\n";
        return 1;
    }

    try {
        loadDataset(filename, X, y); // Load the selected dataset
    } catch (const runtime_error& e) {
        cerr << e.what() << endl;
        return 1;
    }

    cout << "Welcome to Feature Selection Algorithm!" << endl;
    int totalFeatures = X[0].size(); // Set total features dynamically based on the dataset
    cout << "Total number of features: " << totalFeatures << endl;

    cout << "\nType the number of the algorithm you want to run:\n";
    cout << "1. Forward Selection\n";
    cout << "2. Backward Elimination\n";
    cout << "3. Randomized Search\n";
    cout << "Enter your choice (1, 2, or 3): ";
    int algorithm_choice;
    cin >> algorithm_choice;

    if (algorithm_choice == 1) {
        forwardSelection(totalFeatures, X, y);
    } else if (algorithm_choice == 2) {
        backwardElimination(totalFeatures, X, y);
    } else if (algorithm_choice == 3) {
        cout << "Enter the number of iterations for randomized search: ";
        int maxIterations;
        cin >> maxIterations;
        randomizedSearch(totalFeatures, X, y, maxIterations);
    } else {
        cerr << "Invalid algorithm choice. Exiting program.\n";
        return 1;
    }

    return 0;
}
