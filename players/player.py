from abc import ABC, abstractmethod


class Player(ABC):
    game = None

    @abstractmethod
    def take_action(self):
        pass

    def reward(self, _reward):
        pass

    def game_over(self):
        pass

    @classmethod
    def __str__(cls):
        return cls.__name__
