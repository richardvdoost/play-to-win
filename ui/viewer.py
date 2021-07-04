from abc import ABC, abstractmethod

import numpy as np


class Viewer(ABC):
    def __init__(self, game):
        import pygame
        import pygame.gfxdraw

        self.game = game

        self.pygame = pygame
        self.pygame.init()
        self.pygame.display.set_caption(f"Playing {self.game.__class__.__name__}")
        self.screen = self.pygame.display.set_mode(self.screen_size)
        self.clock = self.pygame.time.Clock()
        self.exited = False

        self.init_ui()

    @abstractmethod
    def init_ui(self):
        ...

    @abstractmethod
    def render(self, *args):
        ...


class BoardGameViewer(Viewer):
    background_color = (255, 255, 255)
    line_color = (0, 0, 0)
    stone_colors = ((0, 0, 0), (255, 255, 255), (240, 16, 16), (16, 240, 16), (16, 16, 240))
    semi_transparent_opacity = 160
    line_width = 3
    outer_line_width = 5
    star_point_radius = 3

    mouse_was_pressed = False

    def init_ui(self):
        self.player_colors = [
            self.stone_colors[i % len(self.stone_colors)] for i in range(self.game.player_count)
        ]

        self.line_positions = [
            [
                self.game.border_space + self.game.grid_size / 2 + line_index * (self.game.grid_size + 1)
                for line_index in range(line_count)
            ]
            for line_count in self.game.board_shape
        ]

    def render(self, ghost_stone=None, action_probabilities=None):
        if self.exited:
            return

        # Reset screen
        self.screen.fill(self.background_color)

        # Draw lines
        for dimension, line_count in enumerate(self.game.board_shape):
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
        for star_point in self.game.star_points:
            x, y = self.row_col_to_x_y(*star_point)

            self.pygame.gfxdraw.aacircle(self.screen, x, y, self.star_point_radius, self.line_color)
            self.pygame.gfxdraw.filled_circle(self.screen, x, y, self.star_point_radius, self.line_color)

        # Draw stones based on state
        for i in range(self.game.board_shape[0]):
            for j in range(self.game.board_shape[1]):
                player_index = self.game.state[i, j]

                if player_index == -1:
                    continue

                self.draw_stone(i, j, self.player_colors[player_index % len(self.player_colors)])

        if ghost_stone:
            self.draw_stone(
                *ghost_stone,
                self.player_colors[self.game.current_player],
                self.semi_transparent_opacity,
            )

        # Draw action probabilities if needed
        if action_probabilities is not None:
            self.draw_action_probabilities(action_probabilities)

        # Update display
        self.pygame.display.flip()
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.pygame.quit()
                break

    def draw_stone(self, i, j, color, alpha=255):
        x, y = self.row_col_to_x_y(i, j)
        radius = self.game.grid_size // 2
        self.pygame.gfxdraw.filled_circle(
            self.screen, x, y, radius, (color[0], color[1], color[2], alpha)
        )
        self.pygame.gfxdraw.aacircle(self.screen, x, y, radius, (0, 0, 0, alpha))
        self.pygame.gfxdraw.aacircle(self.screen, x, y, radius - 1, (0, 0, 0, alpha))

    def draw_action_probabilities(self, action_probabilities):
        action_probabilities = action_probabilities.reshape(self.game.board_shape)

        for i in range(self.game.board_shape[0]):
            for j in range(self.game.board_shape[1]):
                x, y = self.row_col_to_x_y(i, j)
                size = np.sqrt(action_probabilities[i, j]) * self.game.grid_size
                color = (64, 192, 32, 192)
                self.pygame.gfxdraw.box(
                    self.screen,
                    [int(x - size / 2), int(y - size / 2), size, size],
                    color,
                )

    def row_col_to_x_y(self, i, j):
        return int(round(self.line_positions[1][j])), int(round(self.line_positions[0][i]))

    def x_y_to_row_col(self, x, y):
        row, col = None, None
        border_space = self.game.border_space
        grid_size = self.game.grid_size

        if border_space < y and y < border_space + self.game.board_shape[0] * grid_size:
            row = int((y - border_space) / grid_size)

        if border_space < x and x < border_space + self.game.board_shape[1] * grid_size:
            col = int((x - border_space) / grid_size)

        return row, col

    def get_pygame_move(self):
        x, y = self.pygame.mouse.get_pos()
        i, j = self.x_y_to_row_col(x, y)

        if i is None or j is None:
            self.render()
            return None

        action = i * self.game.board_shape[0] + j
        if action in self.game.legal_actions:
            self.render(ghost_stone=(i, j))

            mouse_is_pressed, *_ = self.pygame.mouse.get_pressed()
            if not self.mouse_was_pressed and mouse_is_pressed:
                print("Pressed Mouse!")
                self.mouse_was_pressed = True

            if self.mouse_was_pressed and not mouse_is_pressed:
                print("Released Mouse!")
                self.mouse_was_pressed = False
                return action

        else:
            self.render()

        return None

    @property
    def screen_size(self):
        return tuple(
            [
                self.game.border_space * 2 + dimension * self.game.grid_size
                for dimension in self.game.board_shape[::-1]
            ]
        )
