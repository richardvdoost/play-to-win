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
        self.__logit = np.zeros(shape)
        self.__output = np.zeros(shape)
        self.__delta = np.zeros(shape)

        # Set these later
        self.synapses_right = self.synapses_left = None

    def forward_prop(self):
        self.synapses_left.forward_prop()
        self.activate()

    def activate(self):
        self.output = self.__activation_function.activate(self.logit)

    def back_prop(self):
        self.synapses_right.calculate_gradients()

        if self.synapses_left is not None:
            self.gradient()

    def gradient(self):
        self.delta = self.neurons_right.delta.dot(self.synapses_right.weights)[
            :, 1:
        ] * self.__activation_function.gradient(self.logit)

    @property
    def logit(self):
        return self.__logit

    @property
    def output(self):
        return self.__output

    @property
    def output_with_bias(self):
        bias_col = np.ones((self.output.shape[0], 1))
        return np.append(bias_col, self.output, axis=1)

    @property
    def delta(self):
        return self.__delta

    @property
    def neurons_right(self):
        return self.synapses_right.neurons_right if self.synapses_right is not None else None

    @logit.setter
    def logit(self, logit):
        assert logit.shape[1] == self.__length
        self.__logit = logit

    @output.setter
    def output(self, output):
        assert output.shape[1] == self.__length
        self.__output = output

    @delta.setter
    def delta(self, delta):
        assert delta.shape[1] == self.__length
        self.__delta = delta

    def __len__(self):
        return self.__length
