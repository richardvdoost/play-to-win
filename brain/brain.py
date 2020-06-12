import numpy as np
from .neuron_layer import NeuronLayer
from .synapse_cluster import SynapseCluster


class Brain:
    def __init__(self, topology):

        # Initialize target output Y (store X and H in layers)
        self.Y = np.zeros((1, topology[-1][0]))

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

    def train(self, data, iteration_count):

        print(f"\n\n== Training batch of {len(data)} training samples in {iteration_count} iterations ==\n")

        # Convert training data to NumPy matrices
        self.X = np.array([sample["x"] for sample in data])
        self.Y = np.array([sample["y"] for sample in data])

        # Start the training
        for i in range(iteration_count):
            self.forward_prop()
            self.back_prop()

            cost = self.cost()
            if i % max(1, iteration_count // 20) == 0:
                print("cost:", round(cost, 3))

            self.optimize_weights()

        print("\nY:")
        print(self.Y)
        print("\nH:")
        print(self.H)

    def forward_prop(self):
        for layer in self.neuron_layers[1:]:
            layer.forward_prop()

    def back_prop(self):
        self.output_layer.Delta = self.H - self.Y
        for layer in reversed(self.neuron_layers[:-1]):
            layer.back_prop()

    def optimize_weights(self):
        for synapse_cluster in self.synapse_clusters:
            synapse_cluster.optimize_weights()

    def cost(self):
        J = -1 * self.Y * np.log(self.H) - (1 - self.Y) * np.log(1 - self.H)
        return np.sum(J) / self.Y.shape[0]

    def validate_weight_gradients(self):
        print("\n\n== Validating gradients ==\n")

        self.forward_prop()
        self.back_prop()

        nudge_size = 1e-4
        for cluster_i, cluster in enumerate(self.synapse_clusters):
            print(f"Synapse cluster #{cluster_i + 1}")

            rows, cols = cluster.weights.shape
            for i in range(rows):
                for j in range(cols):

                    # Nudge the weight up and down and calculate the cost
                    orig_weight = cluster.weights[i, j]
                    costs = []
                    for nudge in (-nudge_size, nudge_size):
                        cluster.weights[i, j] = orig_weight + nudge
                        self.forward_prop()
                        costs.append(self.cost())

                    est_gradient = (costs[1] - costs[0]) / (2 * nudge_size)
                    diff_percent = abs(
                        (est_gradient - cluster.weight_gradients[i, j]) / max(1e8, cluster.weight_gradients[i, j]) * 100
                    )

                    print(
                        f"  weight[{i},{j}]: {cluster.weights[i, j]:7.3f} - gradient: {cluster.weight_gradients[i, j]:9.6f} - validation: {est_gradient:9.6f} - difference: {diff_percent:9.6f}%"
                    )

                    # Restore the original weight! 😱
                    cluster.weights[i, j] = orig_weight

            print()

    @property
    def input_layer(self):
        return self.neuron_layers[0]

    @property
    def output_layer(self):
        return self.neuron_layers[-1]

    @property
    def X(self):
        return self.input_layer.A

    @property
    def H(self):
        return self.output_layer.A

    @X.setter
    def X(self, X):
        self.input_layer.A = X
