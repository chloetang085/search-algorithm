#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

// Declare the evaluate function (to be used in other files)
double evaluate(const vector<int>& featureSet) {
    return 25.0 + static_cast<double>(rand()) / RAND_MAX * 60.0; // Random accuracy between 25% and 85%
}

// Function prototypes
void forwardSelection(int totalFeatures);
void backwardElimination(int totalFeatures);

int main() {
    srand(time(0)); // Seed random number generator

    cout << "Welcome to Feature Selection Algorithm!" << endl;
    cout << "Please enter the total number of features: ";
    int totalFeatures;
    cin >> totalFeatures;

    cout << "\nType the number of the algorithm you want to run:\n";
    cout << "1. Forward Selection\n";
    cout << "2. Backward Elimination\n";
    cout << "3. Special Algorithm (Not implemented)\n";
    cout << "Enter your choice: ";

    int choice;
    cin >> choice;

    if (choice == 1) {
        forwardSelection(totalFeatures);
    } else if (choice == 2) {
        backwardElimination(totalFeatures);
    } else {
        cout << "Special Algorithm is not implemented yet.\n";
    }

    return 0;
}
