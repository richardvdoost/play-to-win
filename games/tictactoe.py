import numpy as np
import pygame

from .game import Game


class TicTacToe(Game):
    board_shape = (3, 3)
    star_points = ((1, 1),)
    grid_size = 72
    border_space = 16
    mouse_was_pressed = False

    def apply_action(self, action):
        assert np.count_nonzero(action) == 1  # Allow only one action
        assert self.state[action] == -1  # Allow only an action on an empty cell
        self.state[action] = self.active_player_index
        self.last_played_action = action

    def get_pygame_action(self):

        x, y = pygame.mouse.get_pos()
        i, j = self.x_y_to_row_col(x, y)

        if i is None or j is None:
            self.render()
            return None

        if self.allowed_actions[i, j]:
            self.render((i, j))

            mouse_is_pressed, *_ = pygame.mouse.get_pressed()
            if not self.mouse_was_pressed and mouse_is_pressed:
                self.mouse_was_pressed = True

            if self.mouse_was_pressed and not mouse_is_pressed:
                self.mouse_was_pressed = False
                action = np.zeros(self.allowed_actions.shape, dtype=bool)
                action[i, j] = True
                return action

        else:
            self.render()

        return None

    def has_winner(self):

        # Check rows
        for row in range(3):

            # This row has an empty spot? No winner, continue to next row
            if np.any(self.state[row, :] == -1):
                continue

            # This row has 3 similar spots? Winner
            player_index = self.state[row, 0]
            if np.all(self.state[row, 1:] == player_index):
                self.set_winner(player_index)
                return True

        # Check colums
        for col in range(3):

            # This column has an empty spot? No winner, continue to next column
            if np.any(self.state[:, col] == -1):
                continue

            # This column has 3 similar spots? Winner
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
