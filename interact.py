from games import TicTacToe
from players import RandomPlayer, HumanPlayer

human = HumanPlayer()
robot = RandomPlayer()

game = TicTacToe((human, robot))

game.play(1, render=True)
