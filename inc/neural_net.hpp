#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ad_core.hpp"

struct Neuron {
    std::vector<Parameter> weights;
    Parameter bias;

    // The RNG is passed in rather than kept in a function-local static: a shared
    // static is neither thread-safe nor seedable, which made training runs
    // impossible to reproduce.
    Neuron(std::size_t num_inputs, std::mt19937& gen);

    template <typename T>
    ADouble forward(const std::vector<T>& inputs, bool is_output) {
        if (inputs.size() != weights.size()) {
            throw std::invalid_argument("Neuron::forward: expected " +
                                        std::to_string(weights.size()) + " inputs but got " +
                                        std::to_string(inputs.size()));
        }
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

    Layer(std::size_t num_neurons, std::size_t num_inputs, bool is_output, std::mt19937& gen);

    template <typename T>
    std::vector<ADouble> forward(const std::vector<T>& inputs) {
        std::vector<ADouble> outputs;
        outputs.reserve(neurons.size());
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

    // `topology` must name at least an input and an output size, and every entry
    // must be positive. Reproducible when constructed with an explicit seed.
    explicit NeuralNetwork(const std::vector<int>& topology, std::uint32_t seed);
    explicit NeuralNetwork(const std::vector<int>& topology);  // nondeterministic seed

    std::size_t input_size() const;

    template <typename T>
    std::vector<ADouble> forward(const std::vector<T>& inputs) {
        // The constructor guarantees layers is non-empty.
        std::vector<ADouble> current = layers[0].forward(inputs);
        for (std::size_t i = 1; i < layers.size(); ++i) {
            current = layers[i].forward(current);
        }
        return current;
    }

    void update(double lr);
    void update_adam(double lr, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
};

// Mean squared error: the *mean* over predictions, so the gradient magnitude no
// longer scales with the number of outputs. Throws when the sizes disagree.
ADouble mse_loss(const std::vector<ADouble>& preds, const std::vector<double>& targets);

// Sum of squared errors, for callers that genuinely want the unnormalised sum.
ADouble sse_loss(const std::vector<ADouble>& preds, const std::vector<double>& targets);
