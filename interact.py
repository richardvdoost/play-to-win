from games import TicTacToe
from brain import Brain
from players import PolicyGradientPlayer, HumanPlayer, RandomPlayer
from brain.activation_functions import ReLU, Softmax

human = HumanPlayer()

DISCOUNT_FACTOR = 0.6
REWARD_FACTOR = 2
EXPERIENCE_BATCH_SIZE = 512
BATCH_ITERATIONS = 128
EXPERIENCE_BUFFER_SIZE = 2 ** 15

LEARNING_RATE = 0.0005
REGULARIZATION = 0.1

BRAIN_TOPOLOGY = (
    (18, None),
    (512, ReLU),
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
    batch_iterations=1,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)

human_game = TicTacToe((human, robot))
training = TicTacToe((robot, train_player))
random_training = TicTacToe((robot, RandomPlayer()))

robot.act_greedy = True
robot.show_action_probabilities = True

playing = True
while playing:

    # Gain experience, no learning to keep it fast
    robot.learn_while_playing = False
    random_training.play(32)
    training.play(32)

    # Learn on every move of the human game
    robot.learn_while_playing = True
    playing = human_game.play(2, render=True, pause=0.5)
