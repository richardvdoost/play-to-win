from abc import ABC, abstractmethod

import numpy as np


class Game(ABC):

    background_color = (255, 255, 255)
    line_color = (0, 0, 0)
    stone_colors = ((0, 0, 0), (255, 255, 255), (240, 16, 16), (16, 240, 16), (16, 16, 240))
    line_width = 1
    outer_line_width = 3
    star_point_radius = 2

    def __init__(self, players):
        """
        Args:
            players : (Player, Player)
            The two players that will be playing this game
        """

        self.players = players
        self.player_count = len(self.players)

        self.state = None
        self.active_player_index = 0
        self.last_played_action = None
        self.player_colors = [
            self.stone_colors[i % len(self.stone_colors)] for i in range(self.player_count)
        ]

        self.pygame = None
        self.screen = None
        self.clock = None

        self.line_positions = [
            [
                self.border_space + self.grid_size / 2 + line_index * (self.grid_size + 1)
                for line_index in range(line_count)
            ]
            for line_count in self.board_shape
        ]

        self.reset_score()

    def init_pygame(self):
        import pygame
        import pygame.gfxdraw

        self.pygame = pygame
        self.pygame.init()
        self.pygame.display.set_caption(f"Playing {self.__class__.__name__}")
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = self.pygame.time.Clock()

    def render(self, ghost_stone=None, action_probabilities=None):

        # Reset screen
        self.screen.fill(self.background_color)

        # Draw lines
        for dimension, line_count in enumerate(self.board_shape):
            for line_index in range(line_count):
                line_width = (
                    self.outer_line_width
                    if line_index == 0 or line_index == line_count - 1
                    else self.line_width
                )
                start = [
                    self.line_positions[coordinate][line_index if dimension == coordinate else 0]
                    - ((line_width - 1) / 2 if dimension != coordinate else 0)
                    for coordinate in range(1, -1, -1)
                ]
                end = [
                    self.line_positions[coordinate][line_index if dimension == coordinate else -1]
                    + ((line_width - 1) / 2 if dimension != coordinate else 0)
                    for coordinate in range(1, -1, -1)
                ]
                self.pygame.draw.line(
                    self.screen,
                    self.line_color,
                    tuple(start),
                    tuple(end),
                    line_width,
                )

        # Draw star points
        for star_point in self.star_points:
            x, y = self.row_col_to_x_y(*star_point)

            self.pygame.gfxdraw.aacircle(self.screen, x, y, self.star_point_radius, self.line_color)
            self.pygame.gfxdraw.filled_circle(self.screen, x, y, self.star_point_radius, self.line_color)

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
        if action_probabilities is not None:
            self.draw_action_probabilities(action_probabilities)

        # Update display
        self.pygame.display.flip()

    def draw_stone(self, i, j, color, alpha=255):
        x, y = self.row_col_to_x_y(i, j)
        radius = int(round(self.grid_size / 2))
        self.pygame.gfxdraw.filled_circle(
            self.screen, x, y, radius, (color[0], color[1], color[2], alpha)
        )
        self.pygame.gfxdraw.aacircle(self.screen, x, y, radius, (0, 0, 0, alpha))

    def draw_action_probabilities(self, action_probabilities):
        action_probabilities = action_probabilities.reshape(self.board_shape)

        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1]):
                x, y = self.row_col_to_x_y(i, j)
                size = np.sqrt(action_probabilities[i, j]) * self.grid_size
                color = (
                    (64, 192, 32, 192)
                    if self.allowed_actions[i, j] or self.last_played_action[i, j]
                    else (192, 64, 32, 192)
                )
                self.pygame.gfxdraw.box(
                    self.screen,
                    [int(x - size / 2), int(y - size / 2), size, size],
                    color,
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
        self.score = [0] * self.player_count

    def play(self, count, render=False, pause=None):

        if render:
            self.init_pygame()

        for i in range(count):
            self.reset_state()
            self.active_player_index = i % self.player_count

            if render:
                self.init_player_colors()
                self.render()

            # Game episode loop
            while True:

                player = self.players[self.active_player_index]
                player.index = (
                    self.active_player_index
                )  # Moved from init because players can play different games

                action = player.take_action(self)

                if action is None:
                    if render:
                        self.pygame.quit()
                    return False

                self.apply_action(action)

                if render:
                    self.render()

                    if pause and player.is_bot:
                        for _ in range(int(round(pause * 60))):
                            for event in self.pygame.event.get():
                                if event.type == self.pygame.QUIT:
                                    self.pygame.quit()
                                    return False
                            self.clock.tick(60)

                if self.has_finished():
                    break

                self.active_player_index = (self.active_player_index + 1) % self.player_count

            if render:
                self.render()

                for _ in range(int((pause if pause is not None else 1) * 60)):
                    for event in self.pygame.event.get():
                        if event.type == self.pygame.QUIT:
                            self.pygame.quit()
                            break
                    self.clock.tick(60)

            for player in self.players:
                player.game_over(self)

        return True

    def init_player_colors(self):
        for i in range(self.player_count):
            self.player_colors[(self.active_player_index + i) % self.player_count] = self.stone_colors[i]

    def has_finished(self):
        return self.has_winner() or not self.allowed_actions.any()

    def reset_state(self):
        self.state = np.zeros(self.board_shape, dtype=int) - 1

    def set_winner(self, winner_index):
        self.score[winner_index] += 1
        self.players[winner_index].reward(1)
        loser_index = (winner_index + 1) % self.player_count
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
        return tuple(
            [self.border_space * 2 + dimension * self.grid_size for dimension in self.board_shape[::-1]]
        )

    @property
    @abstractmethod
    def allowed_actions(self):
        pass
