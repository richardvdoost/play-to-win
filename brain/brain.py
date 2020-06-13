import numpy as np
from .neuron_layer import NeuronLayer
from .synapse_cluster import SynapseCluster


class Brain:
    """
    Artificial Neural Network

    The Brain consists of L NeuronLayers and L-1 SynapseClusters, specified by the topology.

    NeuronLayers are connected through SynapseClusters which hold the individual connections/weights.

    All connections/weights between two NeuronLayers are grouped together in a SynapseCluster so that they can be
    represented by a NumPy matrix.
    """

    def __init__(self, topology):
        """
        Args:
            topology ((int, TransferFunction|None), ...): Describes each layer of the network with a lenght for the
                layer (amount of neurons) and a transfer function object for the neurons in the layer. The input layer
                can't have a transfer function, so providing None is fine. The output layer should have a ReLU transfer
                function for weight gradient calculation to work properly with the current cost function.
        """

        # Create the neuron layers and synapse clusters (connections)
        self.neuron_layers = []
        self.synapse_clusters = []
        for layer_length, activation_function in topology:

            self.neuron_layers.append(NeuronLayer(layer_length, activation_function))

            # Create a synapse cluster only after we have more than one neuron layers
            if len(self.neuron_layers) > 1:

                surrounding_neuron_layers = (
                    self.neuron_layers[-2],
                    self.neuron_layers[-1],
                )

                self.synapse_clusters.append(SynapseCluster(surrounding_neuron_layers))

        # Initialize the output target (Y)
        self.target = np.zeros((1, topology[-1][0]))

    def train(self, training_samples, iteration_count):
        """
        Train the network on some training samples for a number of iterations
        
        Args:
            data (tuple): Tuple of dictionaries with an 'x' and 'y' key that hold training data
            iteration_count (int): Numer of iterations to run on the training data
        """

        print(f"\n\n== Training batch of {len(training_samples)} training samples in {iteration_count} iterations ==\n")

        self.convert_training_samples(training_samples)

        # Start the training iterations
        for i in range(iteration_count):
            self.forward_prop()
            self.back_prop()

            # Print the progress (cost) 20 times during the training
            if i % max(1, iteration_count // 20) == 0:
                cost = self.cost()
                print("cost:", round(cost, 3))

            self.optimize_weights()

        # Print the end result
        print("\ntarget:")
        print(self.target)
        print("\noutput:")
        print(self.output)

    def convert_training_samples(self, training_samples):
        """ Convert training data to NumPy matrices and initialize the input (X) and target (Y) of the network """

        assert len(training_samples[0]["input"]) == len(self.input_layer)
        assert len(training_samples[0]["target"]) == len(self.output_layer)

        self.input = np.array([sample["input"] for sample in training_samples])
        self.target = np.array([sample["target"] for sample in training_samples])

    def forward_prop(self):
        for layer in self.neuron_layers[1:]:
            layer.forward_prop()

    def back_prop(self):
        self.output_layer.delta = self.output - self.target  # Only works for sigmoid layers
        for layer in reversed(self.neuron_layers[:-1]):
            layer.back_prop()

    def optimize_weights(self):
        for synapse_cluster in self.synapse_clusters:
            synapse_cluster.optimize_weights()

    def cost(self):
        J = -1 * self.target * np.log(self.output) - (1 - self.target) * np.log(1 - self.output)
        return np.sum(J) / self.target.shape[0]

    @property
    def input_layer(self):
        return self.neuron_layers[0]

    @property
    def output_layer(self):
        return self.neuron_layers[-1]

    @property
    def input(self):
        return self.input_layer.output

    @property
    def output(self):
        return self.output_layer.output

    @input.setter
    def input(self, input):
        self.input_layer.output = input
