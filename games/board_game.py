from abc import ABC, abstractmethod

import numpy as np
from ui.viewer import BoardGameViewer


class BoardGame(ABC):
    EMPTY = -1

    def __init__(self):
        self.current_player = None
        self.rewards = [0] * self.player_count

        self.viewer = None

        self.reset()

    def reset(self):
        self.state = np.zeros(self.board_shape, np.uint8) + self.EMPTY
        self.rewards = [0] * self.player_count
        self.current_player = 0

    def get_reward(self, player_id=None):
        """Return the player reward and reset it back to 0"""
        player_id = player_id if player_id is not None else self.current_player
        reward = self.rewards[player_id]
        self.rewards[player_id] = 0

        return reward

    def apply_move(self, action):
        print(f"BoardGame: apply_move({action}) by player {self.current_player}")

        if not action in self.legal_actions:
            print(" - Illegal action!")
            self.rewards[self.current_player] = -1
            return True

        self.change_state(action)

        done = self.is_finished()

        if not done:
            self.current_player = (self.current_player + 1) % self.player_count

        if self.viewer is not None:
            self.viewer.render()

        return done

    def is_finished(self):
        if self.is_winner():
            self.reward_and_punish()
            return True

        return self.is_draw()

    def reward_and_punish(self):
        self.rewards = [1 if id == self.current_player else -1 for id in range(self.player_count)]

    def render_console(self):
        print(self.state + 1)

    def render_viewer(self, action_probabilities=None):
        if self.viewer is None:
            self.viewer = BoardGameViewer(self)
        self.viewer.render(action_probabilities=action_probabilities)

    @abstractmethod
    def change_state(self, action):
        raise NotImplementedError

    @abstractmethod
    def is_draw(self):
        raise NotImplementedError

    @abstractmethod
    def is_winner(self):
        raise NotImplementedError

    @property
    def legal_actions(self):
        raise NotImplementedError
