from .player import Player


class HumanPlayer(Player):
    is_bot = False

    def take_action(self, game):
        while True:

            for event in game.pygame.event.get():
                if event.type == game.pygame.QUIT:
                    game.pygame.quit()
                    return None

            action = game.get_pygame_action()

            if action is not None:
                return action

            game.clock.tick(60)

    def game_over(self, game):
        score_string = [f"{player} {game.score[player_index]}" for player_index, player in enumerate(game.players)]
        print(" - ".join(score_string))

