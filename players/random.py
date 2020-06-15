import numpy as np
from .player import Player


class RandomPlayer(Player):
    def take_action(self, _state, allowed_actions):

        # Find the coordinates of all allowed actions
        allowed_coords = np.argwhere(allowed_actions)
        allowed_coord_count = len(allowed_coords)

        assert allowed_coord_count > 0  # Make sure we have at least one allowed action

        # Make a random choice of coordinates
        choice = np.random.choice(allowed_coord_count)

        # Create a new action matrix
        action = np.zeros(allowed_actions.shape, dtype=bool)
        action[tuple(allowed_coords[choice])] = True

        return action
