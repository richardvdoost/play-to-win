from abc import ABC, abstractmethod

import numpy as np
from ui.viewer import BoardGameViewer

RENDER_MODE_CONSOLE = "CONSOLE"
RENDER_MODE_VIEWER = "VIEWER"


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

    def render_console(self):
        print(self.state)

    def render_viewer(self, action_probabilities=None):
        if self.viewer is None:
            self.viewer = BoardGameViewer(self)
        self.viewer.render(action_probabilities=action_probabilities)

    @abstractmethod
    def apply_move(self, action):
        raise NotImplementedError

    @abstractmethod
    def is_finished(self):
        raise NotImplementedError

    @property
    def legal_actions(self):
        raise NotImplementedError
