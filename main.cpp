#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include "FeatureSelection.h" 
#include "Validator.h"
#include "NNClassifier.h"

using namespace std;

// Group: Chloe Tang – 862337626 – Huong Le – 862388494
// Small Dataset Results:
// Forward: Feature Subset: {1}, Acc: 0.75
// Backward: Feature Subset: {6} Acc: 0.75
// Random: Feature Subset: {3} Acc: 0.68
// Large Dataset Results:
// Forward: Feature Subset: {28,2}, Acc: 0.95
// Backward: Feature Subset: {1,27}, Acc: 0.95

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

    cout << "Dataset loaded successfully from " << filename << ".\n";
}

// Main function
int main() {
    srand(time(0)); // Seed for random number generation

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
        filename = "small-test-dataset.txt"; //small
    } else if (datasetChoice == 2) {
        filename = "large-test-dataset.txt"; //large
    } else if (datasetChoice == 3) {
        filename = "titanic-clean.txt"; //titanic
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
    cout << "Total number of features: ";
    int totalFeatures = X[0].size(); // Set total features dynamically based on the dataset
    cout << totalFeatures << endl;

    cout << "\nType the number of the algorithm you want to run:\n";
    cout << "1. Forward Selection\n";
    cout << "2. Backward Elimination\n";
    cout << "Enter your choice: ";

    int choice;
    cin >> choice;

    bool useAllFeatures = true;
    vector<int> selectedFeatures;

    if (choice == 1 || choice == 2) {
        cout << "\nWould you like to use all features or select specific ones?\n";
        cout << "1. Use all features\n";
        cout << "2. Select specific features\n";
        cout << "Enter your choice: ";
        int featureChoice;
        cin >> featureChoice;

        if (featureChoice == 2) {
            useAllFeatures = false;
            cout << "Enter the feature indices to use (space-separated, 1-based, end with -1): ";
            int featureIndex;
            while (cin >> featureIndex && featureIndex != -1) {
                if (featureIndex > 0 && featureIndex <= totalFeatures) {
                    selectedFeatures.push_back(featureIndex - 1); // Convert to 0-based indexing
                } else {
                    cerr << "Invalid feature index: " << featureIndex << endl;
                    return 1;
                }
            }
        }

        if (useAllFeatures) {
            if (choice == 1) {
                forwardSelection(totalFeatures, X, y);
            } else if (choice == 2) {
                backwardElimination(totalFeatures, X, y);
            }
        } else {
            NearestNeighborClassifier nnClassifier;
            Validator validator(nnClassifier, X, y, selectedFeatures);
            double accuracy = validator.validate();
            cout << "Using selected feature(s) { ";
            for (int f : selectedFeatures) cout << f + 1 << " "; // Convert back to 1-based indexing for display
            cout << "} the accuracy is: " << fixed << setprecision(2) << accuracy * 100 << "%\n";
        }

    } else {
        cout << "Invalid choice. Exiting program.\n";
    }

    return 0;
}
