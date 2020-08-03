import pygame
from .player import Player


class HumanPlayer(Player):
    def take_action(self, game):
        clock = pygame.time.Clock()

        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

            action = game.get_pygame_action()

            if action is not None:
                return action

            clock.tick(60)

    def game_over(self, game):
        score_string = [f"{player} {game.score[player_index]}" for player_index, player in enumerate(game.players)]
        print(" - ".join(score_string))

