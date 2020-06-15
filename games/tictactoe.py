import numpy as np
from .game import Game


class TicTacToe(Game):
    """
    Good old game of tic tac toe, needs 2 players
    """

    def reset_state(self):
        self.state = np.zeros((3, 3), dtype=int) - 1  # -1 means empty

    def apply_action(self, player_index, action):
        assert np.count_nonzero(action) == 1  # Allow only one action
        assert self.state[action] == -1  # Allow only an action on an empty cell
        self.state[action] = player_index

    def has_winner(self):

        # Check rows
        for row in range(3):

            # Any row has an empty spot? No winner
            if np.any(self.state[row, :] == -1):
                break

            # Any row has 3 similar spots? Winner
            player_index = self.state[row, 0]
            if np.all(self.state[row, 1:] == player_index):
                self.set_winner(player_index)
                return True

        # Check colums
        for col in range(3):

            # Any column has an empty spot? No winner
            if np.any(self.state[:, col] == -1):
                break

            # Any column has 3 similar spots? Winner
            player_index = self.state[0, col]
            if np.all(self.state[1:, col] == player_index):
                self.set_winner(player_index)
                return True

        # Check diagonals
        player_index = self.state[1, 1]
        if player_index == -1:
            return False

        if np.all(self.state[(0, 2), (0, 2)] == player_index) or np.all(self.state[(0, 2), (2, 0)] == player_index):
            self.set_winner(player_index)
            return True

        return False

    def set_winner(self, winner_index):
        self.score[winner_index] += 1
        self.players[winner_index].reward(1)
        loser_index = (winner_index + 1) % len(self.players)
        self.players[loser_index].reward(-1)

    @property
    def allowed_actions(self):
        return self.state == -1
