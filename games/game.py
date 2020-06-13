from abc import ABC, abstractmethod


class Game(ABC):
    def __init__(self, players):
        """
        Args:
            players : (Player, Player)
            The two players that will be playing this game
        """

        self.players = players

        # Give the players a reference to this game
        for player in self.players:
            player.set_game(self)

        self.init_state()

    def play(self, count):

        for i in range(count):
            active_player_index = i % len(self.players)

            while not self.has_finished():
                action = self.players[active_player_index].take_action()
                self.apply_action(active_player_index, action)
                active_player_index = (active_player_index + 1) % len(self.players)

    def has_finished(self):
        return self.has_winner() or not self.allowed_actions.any()

    @abstractmethod
    def init_state(self):
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
