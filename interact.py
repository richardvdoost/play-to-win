from games import TicTacToe
from players import RandomPlayer, HumanPlayer

human = HumanPlayer()
robot = RandomPlayer()
robot_2 = RandomPlayer()

game = TicTacToe((robot_2, robot))

game.play(10, render=True, pause=0.5)
