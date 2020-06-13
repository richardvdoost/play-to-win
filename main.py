from brain import Brain
from brain.activation_functions import Identity, ReLU, Sigmoid

brain_topology = (
    (2, None),
    (4, ReLU),
    (3, Sigmoid),
    (1, Sigmoid),
)

training_data = (
    {"input": (0, 0), "target": (0,)},
    {"input": (1, 0), "target": (1,)},
    {"input": (0, 1), "target": (1,)},
    {"input": (1, 1), "target": (0,)},
)

my_brain = Brain(brain_topology, regularization=0.1)
my_brain.train(training_data, 500)

print(my_brain)
