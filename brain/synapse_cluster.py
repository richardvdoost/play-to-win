import numpy as np


class SynapseCluster:
    """
    Describes a set of synapses (connections) between 2 layers of neurons
    """

    # Adam algorithm parameters, no need to tweak
    ADAM_BETA_1 = 0.9
    ADAM_BETA_2 = 0.999
    ADAM_EPSILON = 1e-8

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

        weight_matrix_shape = (
            len(self.neurons_right),
            len(self.neurons_left) + 1,
        )  # +1 for bias neuron 👍

        self.weights = np.random.randn(*weight_matrix_shape) * np.sqrt(2 / len(self.neurons_left))
        self.weight_deltas = np.zeros(weight_matrix_shape)
        self.weight_gradients = np.zeros(weight_matrix_shape)

        self.adam_m = self.adam_v = self.adam_iterations_count = 0

    def forward_prop(self):
        self.neurons_right.logit = self.neurons_left.output_with_bias.dot(self.weights.T)

    def calculate_gradients(self):
        self.weight_gradients = (
            self.neurons_right.delta.T.dot(self.neurons_left.output_with_bias) / self.batch_size
        )

        if self.regularization_factor is not None:
            self.weight_gradients[:, 1:] += self.regularization_factor * self.weights[:, 1:]

    def optimize_weights(self):
        """
        Use the Adam algorithm to optimize the weights
        """

        self.adam_iterations_count += 1

        self.adam_m = self.ADAM_BETA_1 * self.adam_m + (1 - self.ADAM_BETA_1) * self.weight_gradients
        self.adam_v = self.ADAM_BETA_2 * self.adam_v + (1 - self.ADAM_BETA_2) * self.weight_gradients ** 2

        m_corr = self.adam_m / (1 - np.power(self.ADAM_BETA_1, self.adam_iterations_count)) + (
            1 - self.ADAM_BETA_1
        ) * self.weight_gradients / (1 - np.power(self.ADAM_BETA_1, self.adam_iterations_count))
        v_corr = self.adam_v / (1 - np.power(self.ADAM_BETA_2, self.adam_iterations_count))

        self.weights -= self.learning_rate * m_corr / (np.sqrt(v_corr) + self.ADAM_EPSILON)

    @property
    def batch_size(self):
        return self.__brain.batch_size

    @property
    def learning_rate(self):
        return self.__brain.learning_rate

    @property
    def regularization_factor(self):
        return self.__brain.regularization_factor
