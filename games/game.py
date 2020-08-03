from abc import ABC, abstractmethod
import numpy as np
import pygame
from pygame import gfxdraw
from players import HumanPlayer


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
        for i, player in enumerate(self.players):
            player.index = i
        self.player_count = len(self.players)

        self.state = None
        self.active_player_index = 0
        self.last_played_action = None
        self.player_colors = [self.stone_colors[i % len(self.stone_colors)] for i in range(len(self.players))]

        self.screen = None

        self.line_positions = [
            [int(round(self.border_space + (0.5 + line_index) * self.grid_size)) for line_index in range(line_count)]
            for line_count in self.board_shape
        ]

        self.reset_score()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption(f"Playing {self.__class__.__name__}")

        self.screen = pygame.display.set_mode(self.screen_size)

    def render(self, ghost_stone=None, show_action_probabilities=False):

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

                self.draw_stone(i, j, self.player_colors[player_index % len(self.player_colors)])

        # Draw ghost stone based on human
        if ghost_stone:
            self.draw_stone(*ghost_stone, self.player_colors[self.active_player_index], 160)

        # Draw action probabilities if needed
        if show_action_probabilities:
            self.draw_action_probabilities()

        # Update display
        pygame.display.flip()

    def draw_stone(self, i, j, color, alpha=255):
        x, y = self.row_col_to_x_y(i, j)
        radius = int(round(self.grid_size / 2 - 1))
        gfxdraw.filled_circle(self.screen, x, y, radius, (color[0], color[1], color[2], alpha))
        gfxdraw.aacircle(self.screen, x, y, radius, (8, 8, 8, alpha))

    def draw_action_probabilities(self):
        action_probabilities = self.players[self.active_player_index].brain.output.reshape(self.board_shape)

        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1]):
                x, y = self.row_col_to_x_y(i, j)
                size = action_probabilities[i, j] * self.grid_size
                color = (
                    (64, 192, 32, 192)
                    if self.allowed_actions[i, j] or self.last_played_action[i, j]
                    else (192, 64, 32, 192)
                )
                pygame.gfxdraw.box(
                    self.screen, [int(x - size / 2), int(y - size / 2), size, size], color,
                )

    def row_col_to_x_y(self, i, j):
        return int(round(self.line_positions[1][j])), int(round(self.line_positions[0][i]))

    def x_y_to_row_col(self, x, y):
        row, col = None, None
        if self.border_space < y and y < self.border_space + self.board_shape[0] * self.grid_size:
            row = int((y - self.border_space) / self.grid_size)

        if self.border_space < x and x < self.border_space + self.board_shape[1] * self.grid_size:
            col = int((x - self.border_space) / self.grid_size)

        return row, col

    def reset_score(self):
        self.score = [0] * len(self.players)

    def play(self, count, render=False, pause=None):

        if render:
            clock = pygame.time.Clock()
            self.init_pygame()

        for i in range(count):
            self.reset_state()
            self.active_player_index = i % len(self.players)

            if render:
                self.init_player_colors()
                self.render()

            # Game episode loop
            while True:

                player = self.players[self.active_player_index]

                action = player.take_action(self)

                if action is None:
                    if render:
                        pygame.quit()
                    return False

                self.apply_action(action)

                if render:
                    self.render(show_action_probabilities=player.show_action_probabilities)

                if pause and not isinstance(player, HumanPlayer):
                    for _ in range(int(round(pause * 60))):
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return False
                        clock.tick(60)

                if self.has_finished():
                    break

                self.active_player_index = (self.active_player_index + 1) % len(self.players)

            if render:
                self.render()

                for _ in range(int((pause if pause is not None else 1) * 60)):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            break
                    clock.tick(60)

            for player in self.players:
                player.game_over(self)

        return True

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
    def apply_action(self, action):
        pass

    @abstractmethod
    def has_winner(self):
        pass

    @property
    def screen_size(self):
        return tuple([self.border_space * 2 + dimension * self.grid_size for dimension in self.board_shape[::-1]])

    @property
    @abstractmethod
    def allowed_actions(self):
        pass

