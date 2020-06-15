from abc import ABC, abstractmethod


class Player(ABC):
    # def set_game(self, game):
    #     self.game = game

    @abstractmethod
    def take_action(self):
        pass

    def reward(self, _reward):
        pass

    def game_over(self):
        pass

    def __str__(self):
        pass
