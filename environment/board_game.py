import math

import gym
import numpy as np


class BoardGameEnvironment(gym.Env):
    def __init__(self, game, opponent_cls, go_first=True):
        self.game = game

        observation_shape = tuple([game.player_count] + list(game.board_shape))
        self.observation_space = gym.spaces.Box(0, 1, observation_shape, bool)
        self.action_space = gym.spaces.Discrete(math.prod(game.actions_shape))

        # Create opponents (all the same type for now)
        opponent_count = game.player_count - 1
        self.opponents = []
        for _ in range(opponent_count):
            self.opponents.append(opponent_cls(self.observation_space, self.action_space))

        self.go_first = go_first

    def reset(self):
        self.game.reset()

        if not self.go_first:
            print("BoardGameEnvironment: reset - Initialize board with opponent moves")
            self.let_opponents_act()

        self.go_first = not self.go_first

        return self.state

    def step(self, action):
        player_id = self.game.current_player
        print("BoardGameEnvironment: Main player:", player_id)

        info = {}

        if action is None:
            print(f"Player {player_id} resigns!")
            done = True
            return self.state, self.game.get_reward(player_id), done, info

        done = self.game.apply_move(action)

        if not done:
            done = self.let_opponents_act()

        return self.state, self.game.get_reward(player_id), done, info

    def let_opponents_act(self):
        done = False

        for opponent in self.opponents:
            action = opponent.act(self.state, self.game.get_reward(), done, self.game.legal_actions)

            if action is None:
                break

            done = self.game.apply_move(action)

            if done:
                break

        return done

    def render(self, mode="human"):
        if mode == "human":
            self.game.render_viewer()
        elif mode == "console":
            self.game.render_console()

    @property
    def state(self):
        state = np.zeros(self.observation_space.shape, bool)

        for i in range(self.game.player_count):
            player_index = (self.game.current_player + i) % self.game.player_count
            state[i, :, :] = self.game.state == player_index

        return state
