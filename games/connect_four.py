import numpy as np
import pygame

from .game import Game


class ConnectFour(Game):
    board_shape = (6, 7)
    star_points = ((1, 1), (1, 3), (1, 5), (4, 1), (4, 3), (4, 5))
    grid_size = 64
    border_space = 16
    mouse_was_pressed = False

    def apply_action(self, action):
        assert np.count_nonzero(action) == 1  # Allow only one action

        col = np.argwhere(action)[0, 0]
        row = self.falling_stone_row(col)

        self.state[row, col] = self.active_player_index
        self.last_played_action = action

    def falling_stone_row(self, col):
        empty_spots = np.where(self.state[:, col] == -1)[0]
        return empty_spots[-1] if len(empty_spots) > 0 else 0

    def get_pygame_action(self):

        x, y = pygame.mouse.get_pos()
        i, j = self.x_y_to_row_col(x, y)

        if i is None or j is None:
            self.render()
            return None

        if self.allowed_actions[j]:

            stone_i = self.falling_stone_row(j)
            self.render(ghost_stone=(stone_i, j))

            mouse_is_pressed, *_ = pygame.mouse.get_pressed()
            if not self.mouse_was_pressed and mouse_is_pressed:
                self.mouse_was_pressed = True

            if self.mouse_was_pressed and not mouse_is_pressed:
                self.mouse_was_pressed = False
                action = np.zeros(self.allowed_actions.shape, dtype=bool)
                action[j] = True
                return action

        else:
            self.render()

        return None

    def draw_action_probabilities(self):
        action_probabilities = self.players[self.active_player_index].brain.output.reshape(self.board_shape[1])

        for j in range(self.board_shape[1]):
            i = self.falling_stone_row(j)
            x, y = self.row_col_to_x_y(i, j)
            size = action_probabilities[j] * self.grid_size
            color = (64, 192, 32, 192) if self.allowed_actions[j] or self.last_played_action[j] else (192, 64, 32, 192)
            pygame.gfxdraw.box(
                self.screen, [int(x - size / 2), int(y - size / 2), size, size], color,
            )

    def has_winner(self):
        # print()
        # print(self.state)
        # print()

        # Check rows
        for row in range(self.board_shape[0]):

            # This row's center is not the current player? No winner, continue to next row
            if self.state[row, 3] != self.active_player_index:
                continue

            for j in range(self.board_shape[1] - 3):
                if np.all(self.state[row, j : j + 4] == self.active_player_index):
                    self.set_winner(self.active_player_index)
                    return True

        # Check colums
        for col in range(self.board_shape[1]):

            # This column's center is not the current player? No winner, continue to next column
            if self.state[3, col] != self.active_player_index:
                continue

            for i in range(self.board_shape[0] - 3):
                # print(f"checking col {col} for winners")
                # print(f"player index: {self.active_player_index}")
                # print(f"self.state[i : i + 4, col] => {self.state[i : i + 4, col]}")
                if np.all(self.state[i : i + 4, col] == self.active_player_index):
                    self.set_winner(self.active_player_index)
                    return True
                # else:
                #     print(" - nope")

        # Check diagonals
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):

                if np.all(self.state[i : i + 4, j : j + 4].diagonal() == self.active_player_index) or np.all(
                    np.fliplr(self.state[i : i + 4, j : j + 4]).diagonal() == self.active_player_index
                ):
                    self.set_winner(self.active_player_index)
                    return True

        return False

    @property
    def allowed_actions(self):
        """ Return a 1d array with size 7 """
        return self.state[0, :] == -1
