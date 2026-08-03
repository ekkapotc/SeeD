#pragma once
#include <vector>
#include <cstddef>
#include "ad_core.hpp"

struct Neuron {
    std::vector<Parameter> weights;
    Parameter bias;
    
    Neuron(int num_inputs);

    template <typename T> 
    ADouble forward(const std::vector<T>& inputs, bool is_output) {
        ADouble activation = bias.var();
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            activation = activation + (weights[i].var() * inputs[i]);
        }
        return is_output ? activation : sigmoid(activation);
    }

    void update(double lr);
    void update_adam(double lr, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
};

struct Layer {
    std::vector<Neuron> neurons;
    bool is_output_layer;
    
    Layer(int num_neurons, int num_inputs, bool is_output);
    
    template <typename T>
    std::vector<ADouble> forward(const std::vector<T>& inputs) {
        std::vector<ADouble> outputs;
        for (auto& neuron : neurons) {
            outputs.push_back(neuron.forward(inputs, is_output_layer));
        }
        return outputs;
    }

    void update(double lr);
    void update_adam(double lr, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
};

struct NeuralNetwork {
    std::vector<Layer> layers;
    
    NeuralNetwork(const std::vector<int>& topology);
    
    template <typename T>
    std::vector<ADouble> forward(const std::vector<T>& inputs) {
        // First layer takes type T (e.g., double)
        std::vector<ADouble> current = layers[0].forward(inputs);
        // Subsequent layers always take ADouble
        for (std::size_t i = 1; i < layers.size(); ++i) {
            current = layers[i].forward(current);
        }
        return current;
    }

    void update(double lr);
    void update_adam(double lr, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
};

ADouble mse_loss(const std::vector<ADouble>& preds, const std::vector<double>& targets);