import numpy as np

from games.board_game import BoardGame


class TicTacToe(BoardGame):
    board_shape = (3, 3)
    actions_shape = (3, 3)

    star_points = ((1, 1),)
    grid_size = 72
    border_space = 16

    def change_state(self, action):
        row = action // 3
        col = action % 3
        self.state[row, col] = self.current_player

    def is_draw(self):
        return np.all(self.state != self.EMPTY)

    def is_winner(self):
        for i in range(3):
            if np.all(self.state[i, :] == self.current_player):
                return True

            if np.all(self.state[:, i] == self.current_player):
                return True

        forward = (0, 1, 2)
        if np.all(self.state[forward, forward] == self.current_player):
            return True

        backward = (2, 1, 0)
        if np.all(self.state[forward, backward] == self.current_player):
            return True

        return False

    @property
    def legal_actions(self):
        empty_cells = self.state == self.EMPTY
        return empty_cells.flatten().nonzero()[0]
