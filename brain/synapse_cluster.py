# Describes a set of synapses (connections) between 2 layers of neurons

import numpy as np


class SynapseCluster:
    def __init__(self, surrounding_neuron_layers):

        # References to neighbours
        self.neurons_left, self.neurons_right = surrounding_neuron_layers
        self.neurons_left.synapses_right = self.neurons_right.synapses_left = self

        weight_matrix_shape = (len(self.neurons_right), len(self.neurons_left) + 1)  # +1 for bias neuron 👍

        self.weights = np.random.random_sample(weight_matrix_shape) - 0.5
        self.weight_gradients = np.zeros(weight_matrix_shape)

    def forward_prop(self):
        self.neurons_right.Z = self.neurons_left.A_with_bias.dot(self.weights.T)

    def calculate_gradients(self):
        m = self.neurons_left.A.shape[0]
        self.weight_gradients = self.neurons_right.Delta.T.dot(self.neurons_left.A_with_bias) / m

    def optimize_weights(self):
        self.weights -= self.weight_gradients
