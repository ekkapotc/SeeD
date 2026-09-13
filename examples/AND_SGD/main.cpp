#include <cstddef>
#include <iostream>
#include <vector>

#include "neural_net.hpp"

int main() {
    // Explicit seed: training runs are reproducible.
    NeuralNetwork nn({2, 5, 5, 1}, 20240513u);

    // Standard SGD learning rate. mse_loss now returns the mean rather than the
    // sum, so the gradient no longer scales with the batch size.
    double learning_rate = 0.05;

    std::vector<std::vector<double>> X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<std::vector<double>> Y = {{0.0}, {0.0}, {0.0}, {1.0}};

    std::cout << "Starting Training Loop for the AND Operator (Vanilla SGD)...\n" << std::endl;

    for (int epoch = 0; epoch <= 50000; ++epoch) {
        // 1. Reset the tape once per epoch.
        tape().reset();

        // 2. Accumulate total loss across the entire batch.
        ADouble total_loss{0.0};

        for (std::size_t i = 0; i < X.size(); ++i) {
            std::vector<double> x_in = {X[i][0], X[i][1]};
            std::vector<ADouble> preds = nn.forward(x_in);
            total_loss = total_loss + mse_loss(preds, Y[i]);
        }
        total_loss = total_loss / static_cast<double>(X.size());

        // 3. Backpropagate through the whole batch computational graph.
        tape().compute_adjoints(total_loss.id);

        // 4. Update parameters once per epoch using vanilla SGD.
        nn.update(learning_rate);

        if (epoch % 5000 == 0) {
            std::cout << "Epoch " << epoch << " | Mean Loss: " << total_loss.val << std::endl;
        }
    }

    std::cout << "\nTraining Complete! Evaluating AND Operator:\n" << std::endl;

    for (std::size_t i = 0; i < X.size(); ++i) {
        // Safe to reset here without an intervening update(): Parameter tracks
        // the tape generation it bound to and re-registers automatically.
        tape().reset();

        std::vector<double> x_in = {X[i][0], X[i][1]};
        std::vector<ADouble> preds = nn.forward(x_in);

        std::cout << X[i][0] << " AND " << X[i][1] << " = " << preds[0].val
                  << " (Target: " << Y[i][0] << ")" << std::endl;
    }

    tape().release();
    return 0;
}
