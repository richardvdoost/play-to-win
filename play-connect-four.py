import pickle
from games import ConnectFour
from brain import Brain
from players import PolicyGradientPlayer, HumanPlayer, RandomPlayer
from brain.activation_functions import ReLU, Softmax

human = HumanPlayer()

BRAIN_TOPOLOGY = (
    (84, None),
    (256, ReLU),
    (256, ReLU),
    (7, Softmax),
)

BRAIN_FILEPATH = "brain/saved/connect-four-brain.pickle"


playing = True
while playing:

    try:
        robot_brain = pickle.load(open(BRAIN_FILEPATH, "rb"))
    except Exception:
        robot_brain = Brain(BRAIN_TOPOLOGY)

    robot = PolicyGradientPlayer(robot_brain)
    robot.show_action_probabilities = True

    robot_opponent = PolicyGradientPlayer(robot_brain)
    robot_opponent.show_action_probabilities = True
    # robot_opponent.act_greedy = True

    # human_game = ConnectFour((human, robot))
    robot_game = ConnectFour((robot, robot_opponent))

    # playing = human_game.play(2, render=True, pause=0.5)
    robot_game.play(2, render=True, pause=0.7)
