from abc import ABC, abstractmethod
import numpy as np
import pygame
from pygame import gfxdraw


class Game(ABC):

    background_color = (224, 191, 148)
    line_color = (48, 32, 10)
    line_width = 3
    star_point_radius = 5
    stone_colors = ((8, 8, 8), (240, 240, 240), (240, 16, 16), (16, 240, 16), (16, 16, 240))

    def __init__(self, players):
        """
        Args:
            players : (Player, Player)
            The two players that will be playing this game
        """

        self.players = players

        self.state = None
        self.active_player_index = 0
        self.player_colors = [self.stone_colors[i % len(self.stone_colors)] for i in range(len(self.players))]

        self.screen = None

        self.line_positions = [
            [self.border_space + self.grid_size + line_index * 2 * self.grid_size for line_index in range(line_count)]
            for line_count in self.board_shape
        ]

        self.reset_score()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption(f"Playing {self.__class__.__name__}")

        self.screen = pygame.display.set_mode(self.screen_size)

    def render(self):

        # Reset screen
        self.screen.fill(self.background_color)

        # Draw lines
        for dimension, line_count in enumerate(self.board_shape):
            for line_index in range(line_count):
                start = [
                    self.line_positions[coordinate][line_index if dimension == coordinate else 0]
                    for coordinate in range(1, -1, -1)
                ]
                end = [
                    self.line_positions[coordinate][line_index if dimension == coordinate else -1]
                    for coordinate in range(1, -1, -1)
                ]
                pygame.draw.line(
                    self.screen, self.line_color, tuple(start), tuple(end), self.line_width,
                )

        # Draw star points
        for star_point in self.star_points:
            x, y = self.row_col_to_x_y(*star_point)

            gfxdraw.aacircle(self.screen, x, y, self.star_point_radius, self.line_color)
            gfxdraw.filled_circle(self.screen, x, y, self.star_point_radius, self.line_color)

        # Draw stones based on state
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1]):
                player_index = self.state[i, j]

                if player_index == -1:
                    continue

                self.render_stone(self.screen, i, j, self.player_colors[player_index % len(self.player_colors)])

        # Update display
        pygame.display.flip()

    def render_stone(self, surface, i, j, color):
        x, y = self.row_col_to_x_y(i, j)
        gfxdraw.filled_circle(surface, x, y, self.grid_size - 1, color)
        gfxdraw.aacircle(surface, x, y, self.grid_size - 1, (8, 8, 8))

    def row_col_to_x_y(self, i, j):
        return int(self.line_positions[1][j] + 0.5), int(self.line_positions[0][i] + 0.5)

    def reset_score(self):
        self.score = [0] * len(self.players)

    def play(self, count, render=False, pause=None):

        if render:
            self.init_pygame()

        if pause is not None:
            clock = pygame.time.Clock()

        for i in range(count):
            self.reset_state()
            self.active_player_index = i % len(self.players)
            self.init_player_colors()

            while not self.has_finished():

                if render:
                    self.render()

                action = self.players[self.active_player_index].take_action(self)

                if action is None:
                    return

                self.apply_action(self.active_player_index, action)

                if pause:
                    for _ in range(int(round(pause * 60))):
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return
                        clock.tick(60)

                self.active_player_index = (self.active_player_index + 1) % len(self.players)

            if render:
                self.render()

            if pause:
                for _ in range(int(round(pause * 60))):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                    clock.tick(60)

            for player in self.players:
                player.game_over()

    def init_player_colors(self):
        for i in range(len(self.players)):
            self.player_colors[(self.active_player_index + i) % len(self.players)] = self.stone_colors[i]

    def has_finished(self):
        return self.has_winner() or not self.allowed_actions.any()

    def reset_state(self):
        self.state = np.zeros(self.board_shape, dtype=int) - 1

    def set_winner(self, winner_index):
        self.score[winner_index] += 1
        self.players[winner_index].reward(1)
        loser_index = (winner_index + 1) % len(self.players)
        self.players[loser_index].reward(-1)

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
    def screen_size(self):
        return tuple([self.border_space * 2 + dimension * self.grid_size * 2 for dimension in self.board_shape[::-1]])

    @property
    @abstractmethod
    def allowed_actions(self):
        pass

