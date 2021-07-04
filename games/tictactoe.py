import numpy as np

from games.board_game import BoardGame


class TicTacToe(BoardGame):
    board_shape = (3, 3)
    actions_shape = (3, 3)
    player_count = 2

    star_points = ((1, 1),)
    grid_size = 72
    border_space = 16

    def apply_move(self, action):
        row = action // 3
        col = action % 3

        print(f"BoardGame: apply_move({action}) [{row}, {col}] by player {self.current_player}")

        if not action in self.legal_actions:
            print(" - Illegal action!")
            self.rewards[self.current_player] = -1
            return True

        self.state[row, col] = self.current_player

        done = self.is_finished()

        if not done:
            self.current_player = (self.current_player + 1) % self.player_count

        return done

    def is_finished(self):
        if self.is_current_player_winner():
            for player_id in range(self.player_count):
                self.rewards[player_id] = 1 if player_id == self.current_player else -1
            print("WINNER")
            return True

        if np.all(self.state != self.EMPTY):
            print("DRAW")
            return True

        return False

    def is_current_player_winner(self):
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
