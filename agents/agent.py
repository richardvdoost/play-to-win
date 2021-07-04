from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, observation_space=None, action_space=None, game=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.game = game

    @abstractmethod
    def act(self, observation, reward, done, legal_actions=None):
        raise NotImplementedError
