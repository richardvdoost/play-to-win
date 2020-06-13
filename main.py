from brain import Brain
from brain.activation_functions import Identity, ReLU, Sigmoid, Softplus

brain_topology = (
    (2, None),
    (2, Softplus),
    (1, Sigmoid),
)

training_data = (
    {"input": (0, 0), "target": (0,)},
    {"input": (1, 0), "target": (1,)},
    {"input": (0, 1), "target": (1,)},
    {"input": (1, 1), "target": (0,)},
)

my_brain = Brain(brain_topology)
my_brain.train(training_data, 500)

print(my_brain)
