#include <random>
#include <cstddef>
#include "neural_net.hpp"

Neuron::Neuron(int num_inputs) {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);

    bias.val = dist(gen);
    for (int i = 0; i < num_inputs; ++i) {
        weights.push_back(Parameter{dist(gen)});
    }
}

void Neuron::update(double lr) {
    bias.update(lr);
    for (auto& w : weights) {
        w.update(lr);
    }
}

void Neuron::update_adam(double lr, double beta1, double beta2, double epsilon) {
    bias.update_adam(lr, beta1, beta2, epsilon);
    for (auto& w : weights) {
        w.update_adam(lr, beta1, beta2, epsilon);
    }
}

Layer::Layer(int num_neurons, int num_inputs, bool is_output)
    : is_output_layer(is_output) {
    for (int i = 0; i < num_neurons; ++i) {
        neurons.push_back(Neuron(num_inputs));
    }
}

void Layer::update(double lr) {
    for (auto& neuron : neurons) {
        neuron.update(lr);
    }
}

void Layer::update_adam(double lr, double beta1, double beta2, double epsilon) {
    for (auto& neuron : neurons) {
        neuron.update_adam(lr, beta1, beta2, epsilon);
    }
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology) {
    for (std::size_t i = 1; i < topology.size(); ++i) {
        bool is_out = (i == topology.size() - 1);
        layers.push_back(Layer(topology[i], topology[i - 1], is_out));
    }
}

void NeuralNetwork::update(double lr) {
    for (auto& layer : layers) {
        layer.update(lr);
    }
}

void NeuralNetwork::update_adam(double lr, double beta1, double beta2, double epsilon) {
    for (auto& layer : layers) {
        layer.update_adam(lr, beta1, beta2, epsilon);
    }
}

ADouble mse_loss(const std::vector<ADouble>& preds, const std::vector<double>& targets) {
    ADouble loss(0.0);
    for (std::size_t i = 0; i < preds.size(); ++i) {
        ADouble diff = preds[i] - targets[i];
        ADouble sq = pow(diff, 2.0);
        loss = loss + sq;
    }
    return loss;
}