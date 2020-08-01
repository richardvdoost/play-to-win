import pygame
from .player import Player


class HumanPlayer(Player):
    def take_action(self):
        clock = pygame.time.Clock()

        # Keep looping until we got a valid action from the game, or when the human wants to exit
        while True:

            action = self.game.get_pygame_action()

            if action is not None:
                return action

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

            clock.tick(60)
