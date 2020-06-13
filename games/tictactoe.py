import numpy as np
from .game import Game


class TicTacToe(Game):
    """
    Good old game of tic tac toe, needs 2 players
    """

    def init_state(self):
        self.state = np.zeros((3, 3)) - 1  # -1 means empty

    def apply_action(self, player_index, action):
        coordinates = action.nonzero()
        assert len(coordinates[0]) == 1 and len(coordinates[1]) == 1

        self.state[coordinates] = player_index

    def has_winner(self):

        # Check rows
        for row in range(3):

            # Any row has an empty spot? No winner
            if np.any(self.state[row, :] == -1):
                break

            # Any row has 3 similar spots? Winner
            player_index = self.state[row, 0]
            if np.all(self.state[row, :] == player_index):
                self.winner_index = player_index
                return True

        # Check colums
        for col in range(3):

            # Any column has an empty spot? No winner
            if np.any(self.state[:, col] == -1):
                break

            # Any column has 3 similar spots? Winner
            player_index = self.state[row, 0]
            if np.all(self.state[:, col] == player_index):
                self.winner_index = player_index
                return True

        # Check diagonals
        player_index = self.state[1, 1]
        if player_index == -1:
            return False

        if player_index in (self.state[0, 0], self.state[2, 2]) or player_index in (self.state[0, 2], self.state[2, 0]):
            self.winner_index = player_index
            return True

    @property
    def allowed_actions(self):
        return self.state == -1
