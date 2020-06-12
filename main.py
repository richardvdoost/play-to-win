import numpy as np
from brain import Brain
from brain.activation_functions import Identity, ReLU, Sigmoid

brain_topology = (
    (2, None),
    (3, Sigmoid),
    (1, Sigmoid),
)

training_data = (
    {"x": (0, 0), "y": (0,)},
    {"x": (1, 0), "y": (1,)},
    {"x": (0, 1), "y": (1,)},
    {"x": (1, 1), "y": (0,)},
)

my_brain = Brain(brain_topology)

my_brain.train(training_data, 1000)

my_brain.validate_weight_gradients()
