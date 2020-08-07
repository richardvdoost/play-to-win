from abc import ABC, abstractmethod


class Player(ABC):
    index = None
    show_action_probabilities = False
    is_bot = True

    @abstractmethod
    def take_action(self, game):
        pass

    def reward(self, _reward):
        pass

    def game_over(self, game):
        pass

    @classmethod
    def __str__(cls):
        return cls.__name__
