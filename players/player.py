from abc import ABC, abstractmethod


class Player(ABC):
    @abstractmethod
    def take_action(self, game):
        pass

    def reward(self, _reward):
        pass

    def game_over(self):
        pass

    @classmethod
    def __str__(cls):
        return cls.__name__
