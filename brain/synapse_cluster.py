import numpy as np


class SynapseCluster:
    """
    Describes a set of synapses (connections) between 2 layers of neurons
    """

    def __init__(self, brain, surrounding_neuron_layers):
        """
        Args:
            brain (Brain): Reference to the brain this synapse cluster is part of
            surrounding_neuron_layers (NeuronLayer, NeuronLayer): The neuron layers that surround the current synapse
                cluster on the left and right side respectively.
        """

        self.__brain = brain

        # References to neighbours
        self.neurons_left, self.neurons_right = surrounding_neuron_layers
        self.neurons_left.synapses_right = self.neurons_right.synapses_left = self

        weight_matrix_shape = (len(self.neurons_right), len(self.neurons_left) + 1)  # +1 for bias neuron 👍

        self.weights = np.random.randn(*weight_matrix_shape) * np.sqrt(2 / len(self.neurons_left))
        self.weight_deltas = np.zeros(weight_matrix_shape)
        self.weight_gradients = np.zeros(weight_matrix_shape)

    def forward_prop(self):
        self.neurons_right.logit = self.neurons_left.output_with_bias.dot(self.weights.T)

    def calculate_gradients(self):
        self.weight_gradients = self.neurons_right.delta.T.dot(self.neurons_left.output_with_bias) / self.batch_size

        if self.regularization_factor is not None:
            self.weight_gradients[:, 1:] += self.regularization_factor * self.weights[:, 1:]

    def optimize_weights(self):
        deltas = self.learning_rate * self.weight_gradients

        if self.momentum is not None:
            deltas += self.momentum * self.weight_deltas

        self.weight_deltas = deltas
        self.weights -= self.weight_deltas

    @property
    def batch_size(self):
        return self.__brain.batch_size

    @property
    def learning_rate(self):
        return self.__brain.learning_rate

    @property
    def momentum(self):
        return self.__brain.momentum

    @property
    def regularization_factor(self):
        return self.__brain.regularization_factor
