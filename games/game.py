from abc import ABC, abstractmethod


class Game(ABC):
    def __init__(self, players):
        """
        Args:
            players : (Player, Player)
            The two players that will be playing this game
        """

        self.players = players
        self.state = None
        self.active_player_index = 0

        self.reset_score()

    def reset_score(self):
        self.score = [0] * len(self.players)

    def play(self, count):

        for i in range(count):

            self.reset_state()
            self.active_player_index = i % len(self.players)

            while not self.has_finished():
                action = self.players[self.active_player_index].take_action(self.state, self.allowed_actions)
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
    def apply_action(self, player_index, action):
        pass

    @abstractmethod
    def has_winner(self):
        pass

    @property
    @abstractmethod
    def allowed_actions(self):
        pass
