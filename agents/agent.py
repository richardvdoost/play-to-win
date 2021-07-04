from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    def __init__(
        self,
        observation_space=None,
        action_space=None,
        game=None,
    ):
        self._observation_space = observation_space
        self._action_space = action_space
        self.game = game

        self.observation_space_length = space_length(observation_space)
        self.action_space_length = space_length(action_space)

    @abstractmethod
    def act(self, observation, reward, done, legal_actions=None):
        raise NotImplementedError

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space):
        self.observation_space_length = space_length(space)
        self._observation_space = space

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, space):
        self.action_space_length = space_length(space)
        self._action_space = space


def space_length(space):
    if space is None:
        return 0
    return space.n if hasattr(space, "n") else np.product(space.shape)
