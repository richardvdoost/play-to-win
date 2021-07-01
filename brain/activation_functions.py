# Bunch of different activation functions with gradients that evaluate element-wise on numpy arrays

from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @staticmethod
    @abstractmethod
    def activate():
        pass

    @staticmethod
    @abstractmethod
    def gradient():
        pass


class Identity(ActivationFunction):
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def gradient(x):
        return np.ones(x.shape)


class ReLU(ActivationFunction):
    @staticmethod
    def activate(x):
        return np.fmax(0, x)

    @staticmethod
    def gradient(x):
        return np.heaviside(x, 0.5)


class LeakyReLU(ActivationFunction):
    @staticmethod
    def activate(x):
        return np.fmax(0.1 * x, x)

    @staticmethod
    def gradient(x):
        return np.heaviside(x, 0.5) * 1.1 - 0.1


class Sigmoid(ActivationFunction):
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-1 * x))

    @classmethod
    def gradient(cls, x):
        activate = cls.activate(x)
        return activate * (1 - activate)


class Softplus(ActivationFunction):
    @staticmethod
    def activate(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def gradient(x):
        return 1 / (1 + np.exp(-1 * x))


class Softmax(ActivationFunction):
    @staticmethod
    def activate(x):
        m = x.shape[0]
        x_rel = x - x.max(1).reshape((m, 1))
        t = np.exp(x_rel)
        return t / t.sum(1).reshape((m, 1))

    @staticmethod
    def gradient(x):
        # Only use softmax as output layer, then we don't need this
        pass


# Testing
if __name__ == "__main__":

    test_values = [-5, -1, -0.5, 0, 0.5, 1, 5]

    x = np.array(test_values)
    activation_functions = (Identity, ReLU, Sigmoid)

    for f in activation_functions:
        print("    ", f.__name__, "\b(", x, ") =", f.activate(x))
        print("d/dx", f.__name__, "\b(", x, ") =", f.gradient(x))
        print()

    print(f"    Softmax:\n{Softmax.activate(np.array([[1,2,3,4],[2,3,4,5]]))}")
