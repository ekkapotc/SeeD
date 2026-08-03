#include <iostream>
#include <vector>
#include <cstddef>

#include "neural_net.hpp"

int main() {
    NeuralNetwork nn({2, 5, 5, 1});
    double learning_rate = 0.01;

    std::vector<std::vector<double>> X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<std::vector<double>> Y = {{0.0}, {0.0}, {0.0}, {1.0}};

    std::cout << "Starting Training Loop for the AND Operator (Adam)...\n" << std::endl;

    for (int epoch = 0; epoch <= 5000; ++epoch) {
        // 1. Reset tape once per epoch
        global_tape.reset();
        
        // 2. Accumulate total loss across the entire batch
        ADouble total_loss{0.0};

        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<double> x_in = {X[i][0], X[i][1]};
            std::vector<ADouble> preds = nn.forward(x_in);
            ADouble loss = mse_loss(preds, Y[i]);
            total_loss = total_loss + loss; 
        }

        // 3. Backpropagate through the whole batch computational graph
        global_tape.compute_adjoints(total_loss.id);
        
        // 4. Update parameters ONCE per epoch using Adam
        nn.update_adam(learning_rate);

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << " | Mean Loss: " << (total_loss.val / X.size()) << std::endl;
        }
    }
    
    std::cout << "\nTraining Complete! Evaluating AND Operator:\n" << std::endl;
    
    for (size_t i = 0; i < X.size(); ++i) {
        global_tape.reset(); 
        
        std::vector<ADouble> x_in = {ADouble(X[i][0]), ADouble(X[i][1])};
        std::vector<ADouble> preds = nn.forward(x_in);
        
        std::cout << X[i][0] << " AND " << X[i][1] 
                  << " = " << preds[0].val 
                  << " (Target: " << Y[i][0] << ")" << std::endl;
    }
    
    return 0;
}