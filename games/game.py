from abc import ABC, abstractmethod
import pygame
from pygame import gfxdraw


class Game(ABC):

    background_color = (224, 191, 148)
    line_color = (48, 32, 10)
    line_width = 3
    star_point_radius = 8

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

        self.reset_score()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption(f"Playing {self.__class__.__name__}")

        self.screen = pygame.display.set_mode(self.screen_size)
        self.screen.fill(self.background_color)

        line_positions = [
            [
                self.border_space + self.stone_radius + line_index * 2 * self.stone_radius
                for line_index in range(line_count)
            ]
            for line_count in self.board_shape
        ]

        for dimension, line_count in enumerate(self.board_shape):
            for line_index in range(line_count):
                start = [
                    line_positions[coordinate][line_index if dimension == coordinate else 0]
                    for coordinate in range(1, -1, -1)
                ]
                end = [
                    line_positions[coordinate][line_index if dimension == coordinate else -1]
                    for coordinate in range(1, -1, -1)
                ]
                pygame.draw.line(
                    self.screen, self.line_color, tuple(start), tuple(end), self.line_width,
                )

        for star_point in self.star_points:
            x = int(line_positions[1][star_point[1]] + 0.5)
            y = int(line_positions[0][star_point[0]] + 0.5)

            gfxdraw.aacircle(self.screen, x, y, self.star_point_radius, self.line_color)
            gfxdraw.filled_circle(self.screen, x, y, self.star_point_radius, self.line_color)

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

    @property
    @abstractmethod
    def screen_size(self):
        pass
