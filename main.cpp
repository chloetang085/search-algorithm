#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include "NNClassifier.h"
#include "Validator.h"
#include "cmath"

// Function to load data from a file
void load_data(const std::string& filename, 
               std::vector<std::vector<double>>& X, 
               std::vector<int>& y) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the file");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int label;
        ss >> label;
        y.push_back(label);

        std::vector<double> features;
        double value;
        while (ss >> value) {
            features.push_back(value);
        }
        X.push_back(features);
    }
}

// Function to normalize features (zero mean, unit variance)
void normalize_features(std::vector<std::vector<double>>& X) {
    size_t num_features = X[0].size();
    std::vector<double> mean(num_features, 0.0);
    std::vector<double> stddev(num_features, 0.0);

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
            stddev[i] += pow(row[i] - mean[i], 2);
        }
    }
    for (size_t i = 0; i < num_features; ++i) {
        stddev[i] = sqrt(stddev[i] / X.size());
        if (stddev[i] == 0) {
            stddev[i] = 1.0; // Prevent division by zero
        }
    }

    // Normalize each feature
    for (auto& row : X) {
        for (size_t i = 0; i < num_features; ++i) {
            row[i] = (row[i] - mean[i]) / stddev[i];
        }
    }
}

int main() {
    std::cout << "Select the dataset to use:\n";
    std::cout << "1. Small dataset (small-test-dataset.txt)\n";
    std::cout << "2. Large dataset (large-test-dataset.txt)\n";
    std::cout << "Enter your choice (1 or 2): ";
    int dataset_choice;
    std::cin >> dataset_choice;

    std::string filename;
    if (dataset_choice == 1) {
        filename = "small-test-dataset.txt";
    } else if (dataset_choice == 2) {
        filename = "large-test-dataset.txt";
    } else {
        std::cerr << "Invalid choice. Exiting.\n";
        return 1;
    }

    // Load data
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    load_data(filename, X, y);

    // Measure normalization time
    auto start_normalize = std::chrono::high_resolution_clock::now();
    normalize_features(X);
    auto end_normalize = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> normalize_time = end_normalize - start_normalize;

    std::cout << "Would you like to use all features or select specific ones?\n";
    std::cout << "1. Use all features\n";
    std::cout << "2. Select specific features\n";
    std::cout << "Enter your choice (1 or 2): ";
    int feature_choice;
    std::cin >> feature_choice;

    std::vector<std::vector<double>> X_subset;
    if (feature_choice == 1) {
        X_subset = X;
    } else if (feature_choice == 2) {
        std::cout << "Enter the feature indices to use (space-separated, 0-based, end with -1): ";
        std::vector<int> feature_subset;
        int feature_index;
        while (std::cin >> feature_index && feature_index != -1) {
            feature_subset.push_back(feature_index);
        }

        for (const auto& row : X) {
            std::vector<double> subset_row;
            for (int index : feature_subset) {
                subset_row.push_back(row[index]);
            }
            X_subset.push_back(subset_row);
        }
    } else {
        std::cerr << "Invalid choice. Exiting.\n";
        return 1;
    }

    // Create and train the NN classifier
    NearestNeighborClassifier nn_classifier;

    // Measure accuracy calculation time
    auto start_validation = std::chrono::high_resolution_clock::now();
    Validator validator(nn_classifier, X_subset, y);
    double accuracy = validator.validate();
    auto end_validation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> validation_time = end_validation - start_validation;

    // Output results
    std::cout << "Time for normalization: " << normalize_time.count() << " ms\n";
    std::cout << "Time for accuracy calculation: " << validation_time.count() << " ms\n";
    std::cout << "Best feature set: All features, Accuracy: " << accuracy * 100 << "%\n";

    return 0;
}
