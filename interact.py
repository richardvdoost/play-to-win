from games import TicTacToe
from brain import Brain
from players import PolicyGradientPlayer, HumanPlayer
from brain.activation_functions import ReLU, Softmax

human = HumanPlayer()

DISCOUNT_FACTOR = 0.5
REWARD_FACTOR = 2
EXPERIENCE_BATCH_SIZE = 1024
BATCH_ITERATIONS = 1
EXPERIENCE_BUFFER_SIZE = 2 ** 16

LEARNING_RATE = 0.002
REGULARIZATION = 0.1

BRAIN_TOPOLOGY = (
    (18, None),
    (1024, ReLU),
    (9, Softmax),
)

robot_brain = Brain(BRAIN_TOPOLOGY, learning_rate=LEARNING_RATE, regularization=REGULARIZATION)
robot = PolicyGradientPlayer(
    robot_brain,
    discount_factor=DISCOUNT_FACTOR,
    reward_factor=REWARD_FACTOR,
    batch_iterations=BATCH_ITERATIONS,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)
train_player = PolicyGradientPlayer(
    robot_brain,
    discount_factor=DISCOUNT_FACTOR,
    reward_factor=REWARD_FACTOR,
    batch_iterations=BATCH_ITERATIONS,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)

game = TicTacToe((human, robot))
training = TicTacToe((robot, train_player))

robot.act_greedy = True
while True:

    game.play(2, render=True)

    print(f"Score: {game.score}")

    training.play(200)
