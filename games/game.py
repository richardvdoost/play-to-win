from abc import ABC, abstractmethod
import pygame


class Game(ABC):
    def __init__(self, players):
        """
        Args:
            players : (Player, Player)
            The two players that will be playing this game
        """

        self.players = players
        for player in self.players:
            player.game = self

        self.state = None
        self.active_player_index = 0

        self.screen = None
        self.screen_size = (360, 360)

        self.reset_score()

    def reset_score(self):
        self.score = [0] * len(self.players)

    def play(self, count, render=False):

        if render:
            self.init_pygame()

        for i in range(count):

            self.reset_state()
            self.active_player_index = i % len(self.players)

            while not self.has_finished():

                if render:
                    self.render()

                action = self.players[self.active_player_index].take_action()

                if action is None:
                    print(f"{self.players[self.active_player_index]} requested to end the game")
                    return

                self.apply_action(self.active_player_index, action)
                self.active_player_index = (self.active_player_index + 1) % len(self.players)

            for player in self.players:
                player.game_over()

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)

    def has_finished(self):
        return self.has_winner() or not self.allowed_actions.any()

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def get_pygame_action(self):
        pass

    @abstractmethod
    def apply_action(self, player_index, action):
        pass

    @abstractmethod
    def has_winner(self):
        pass

    @property
    @abstractmethod
    def allowed_actions(self):
        pass
