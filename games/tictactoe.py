import numpy as np

from .game import Game


class TicTacToe(Game):
    board_shape = (3, 3)
    star_points = ((1, 1),)
    grid_size = 42
    border_space = 8

    def apply_action(self, player_index, action):
        assert np.count_nonzero(action) == 1  # Allow only one action
        assert self.state[action] == -1  # Allow only an action on an empty cell
        self.state[action] = player_index

    def get_pygame_action(self):

        # Feedback on mouse position

        # Listen for mouse clicks and if they're valid, return the action

        return None

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

    @property
    def allowed_actions(self):
        return self.state == -1
