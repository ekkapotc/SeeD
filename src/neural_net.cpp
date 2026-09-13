#include "neural_net.hpp"

#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>

Neuron::Neuron(std::size_t num_inputs, std::mt19937& gen) : bias(0.0) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    bias.val = dist(gen);
    weights.reserve(num_inputs);
    for (std::size_t i = 0; i < num_inputs; ++i) {
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

Layer::Layer(std::size_t num_neurons, std::size_t num_inputs, bool is_output, std::mt19937& gen)
    : is_output_layer(is_output) {
    neurons.reserve(num_neurons);
    for (std::size_t i = 0; i < num_neurons; ++i) {
        neurons.push_back(Neuron(num_inputs, gen));
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

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology, std::uint32_t seed) {
    if (topology.size() < 2) {
        throw std::invalid_argument(
            "NeuralNetwork: topology needs at least an input and an output size, got " +
            std::to_string(topology.size()) + " entries");
    }
    for (std::size_t i = 0; i < topology.size(); ++i) {
        if (topology[i] <= 0) {
            throw std::invalid_argument("NeuralNetwork: topology[" + std::to_string(i) +
                                        "] must be positive, got " + std::to_string(topology[i]));
        }
    }

    std::mt19937 gen(seed);
    layers.reserve(topology.size() - 1);
    for (std::size_t i = 1; i < topology.size(); ++i) {
        const bool is_out = (i == topology.size() - 1);
        layers.push_back(Layer(static_cast<std::size_t>(topology[i]),
                               static_cast<std::size_t>(topology[i - 1]), is_out, gen));
    }
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology)
    : NeuralNetwork(topology, std::random_device{}()) {}

std::size_t NeuralNetwork::input_size() const { return layers.front().neurons.front().weights.size(); }

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

static void check_loss_sizes(const std::vector<ADouble>& preds, const std::vector<double>& targets) {
    if (preds.size() != targets.size()) {
        throw std::invalid_argument("loss: " + std::to_string(preds.size()) + " predictions but " +
                                    std::to_string(targets.size()) + " targets");
    }
    if (preds.empty()) {
        throw std::invalid_argument("loss: no predictions");
    }
}

ADouble sse_loss(const std::vector<ADouble>& preds, const std::vector<double>& targets) {
    check_loss_sizes(preds, targets);
    ADouble loss(0.0);
    for (std::size_t i = 0; i < preds.size(); ++i) {
        loss = loss + pow(preds[i] - targets[i], 2.0);
    }
    return loss;
}

ADouble mse_loss(const std::vector<ADouble>& preds, const std::vector<double>& targets) {
    return sse_loss(preds, targets) / static_cast<double>(preds.size());
}
