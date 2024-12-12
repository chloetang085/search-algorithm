#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include "FeatureSelection.h" // Include the header file for feature selection
#include "Validator.h"
#include "NNClassifier.h"

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
        filename = "small-test-dataset.txt"; // Replace with actual small dataset file path
    } else if (datasetChoice == 2) {
        filename = "large-test-dataset.txt"; // Replace with actual large dataset file path
    } else if (datasetChoice == 3) {
        filename = "titanic-clean.txt"; // Replace with actual Titanic dataset file path
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

    if (choice == 1) {
        forwardSelection(totalFeatures, X, y);
    } else if (choice == 2) {
        backwardElimination(totalFeatures, X, y);
    } else {
        cout << "Invalid choice. Exiting program.\n";
    }

    return 0;
}
