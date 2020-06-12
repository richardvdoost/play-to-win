import numpy as np


class NeuronLayer:
    """
    Represents a layer of neurons in the network
    """

    def __init__(self, length, activation_function):
        """
        Args:
            length: Number of neurons in the layer
            activation_function (ActivationFunction, None): Activation function object with a activate(x) and
                gradient(x) methods. If this is the input layer of the network, activation function can be None.
        """

        # Properties that can't be changed after layer creation
        self.__length = length
        self.__activation_function = activation_function

        # Initialize different neuron values we need
        shape = (1, self.__length)
        self.__Z = np.zeros(shape)
        self.__A = np.zeros(shape)
        self.__Delta = np.zeros(shape)

        # Set these later
        self.synapses_right = self.synapses_left = None

    def forward_prop(self):
        self.synapses_left.forward_prop()
        self.activate()

    def activate(self):
        self.A = self.__activation_function.activate(self.Z)

    def back_prop(self):
        self.synapses_right.calculate_gradients()

        if self.synapses_left is not None:
            self.gradient()

    def gradient(self):
        self.Delta = self.neurons_right.Delta.dot(self.synapses_right.weights)[
            :, 1:
        ] * self.__activation_function.gradient(self.Z)

    @property
    def Z(self):
        return self.__Z

    @property
    def A(self):
        return self.__A

    @property
    def A_with_bias(self):
        bias_col = np.ones((self.__A.shape[0], 1))
        return np.append(bias_col, self.__A, axis=1)

    @property
    def Delta(self):
        return self.__Delta

    @property
    def neurons_right(self):
        return self.synapses_right.neurons_right if self.synapses_right is not None else None

    @Z.setter
    def Z(self, Z):
        assert Z.shape[1] == self.__length
        self.__Z = Z

    @A.setter
    def A(self, A):
        assert A.shape[1] == self.__length
        self.__A = A

    @Delta.setter
    def Delta(self, Delta):
        assert Delta.shape[1] == self.__length
        self.__Delta = Delta

    def __len__(self):
        return self.__length
