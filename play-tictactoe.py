import pickle
from games import TicTacToe
from brain import Brain
from players import PolicyGradientPlayer, HumanPlayer, RandomPlayer
from brain.activation_functions import ReLU, Softmax

human = HumanPlayer()

BRAIN_FILEPATH = "brain/saved/tictactoe-brain.pickle"

playing = True
while playing:

    robot_brain = pickle.load(open(BRAIN_FILEPATH, "rb"))

    robot = PolicyGradientPlayer(robot_brain)
    robot.act_greedy = True
    robot.show_action_probabilities = 0.3

    human_game = TicTacToe((human, robot))
    # robot_game = TicTacToe((robot, robot_opponent))

    playing = human_game.play(2, render=True, pause=0.5)
    # robot_game.play(1, render=True, pause=0.7)
