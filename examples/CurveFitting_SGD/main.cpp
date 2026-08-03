#include <iostream>
#include <vector>
#include <cmath>
#include "ad_core.hpp"

int main() {
    // 1. Define the "true" parameters we want the engine to discover
    double true_A = 2.5;
    double true_B = 1.2;
    double true_C = 0.8;

    // 2. Generate clean synthetic dataset (Smoother Domain)
    std::vector<double> X;
    std::vector<double> Y;
    for (double x = 0.0; x <= 5.0; x += 0.1) {
        X.push_back(x);
        // Calculate true Y values
        Y.push_back(true_A * std::sin(true_B * x + true_C));
    }

    // 3. Initialize our trainable parameters within the basin of attraction
    Parameter A(2.0);
    Parameter B(1.0);
    Parameter C(1.0);

    // Vanilla SGD needs a smaller learning rate and more epochs than Adam
    double learning_rate = 0.001; 
    int epochs = 15000;

    std::cout << "--- Starting Non-Linear Curve Fitting (Vanilla SGD) ---" << std::endl;
    std::cout << "Target Parameters  : A=" << true_A << ", B=" << true_B << ", C=" << true_C << "\n\n";

    // 4. Training Loop
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        global_tape.reset();
        
        ADouble total_loss(0.0);

        // Accumulate loss across the batch
        for (size_t i = 0; i < X.size(); ++i) {
            ADouble y_pred = A.var() * sin(B.var() * X[i] + C.var());
            
            ADouble diff = y_pred - Y[i];
            total_loss = total_loss + pow(diff, 2.0);
        }

        // Average the loss to stabilize gradients regardless of dataset size
        total_loss = total_loss / static_cast<double>(X.size());

        // Backpropagate
        global_tape.compute_adjoints(total_loss.id);

        // Apply Gradients using Vanilla SGD
        A.update(learning_rate);
        B.update(learning_rate);
        C.update(learning_rate);

        // Print progress
        if (epoch % 1500 == 0) {
            std::cout << "Epoch " << epoch 
                      << " | Mean Loss: " << total_loss.val 
                      << " | Current Guess: A=" << A.val 
                      << ", B=" << B.val 
                      << ", C=" << C.val << std::endl;
        }
    }

    std::cout << "\n--- Training Complete ---" << std::endl;
    std::cout << "Final Discovered Equation: y = " 
              << A.val << " * sin(" << B.val << " * x + " << C.val << ")" << std::endl;

    return 0;
}
