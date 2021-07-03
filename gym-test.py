import math
import random

import gym
import numpy as np

EPISODE_COUNT = 1
MAX_TIMESTEPS = 1000
RENDER = True


class BoardGame:
    EMPTY = -1

    board_shape = (3, 3)
    actions_shape = (3, 3)
    player_count = 2

    def __init__(self):
        self.current_player = None
        self.rewards = [0] * self.player_count
        self.reset()

    def reset(self):
        print("BoardGame: reset")
        self.state = np.zeros(self.board_shape, int) + self.EMPTY
        self.rewards = [0] * self.player_count
        self.current_player = 0

    def move(self, action):
        row = action // 3
        col = action % 3
        print(f"BoardGame: move {action} [{row}, {col}] by player {self.current_player}")

        if not action in self.legal_actions:
            print(" - Illegal move (non empty cell)!")
            self.rewards[self.current_player] = -1
            return True

        self.state[row, col] = self.current_player
        self.current_player = (self.current_player + 1) % self.player_count

        return self.is_finished()

    def move_is_allowed(self, row, col):
        if self.state[row, col] != self.EMPTY:
            print(" - Illegal move (non empty cell)!")
            self.rewards[self.current_player] = -1
            return False

        return True

    def is_finished(self):
        if len((self.state == self.EMPTY).flatten().nonzero()) == 0:
            self.rewards[self.current_player] = 1
            return True

        return False

    def get_reward(self):
        reward = self.rewards[self.current_player]
        self.rewards[self.current_player] = 0
        return reward

    def render(self):
        print("BoardGame: render")
        print(self.state + 1)

    @property
    def legal_actions(self):
        empty_cells = self.state == self.EMPTY
        return empty_cells.flatten().nonzero()[0]


class BoardGameEnvironment(gym.core.Env):
    def __init__(self, game, opponent_cls, go_first=True):
        self.game = game

        observation_shape = tuple([game.player_count] + list(game.board_shape))
        self.observation_space = gym.spaces.Box(0, 1, observation_shape, bool)
        self.action_space = gym.spaces.Discrete(math.prod(game.actions_shape))

        opponent_count = game.player_count - 1
        self.opponents = [opponent_cls(self.observation_space, self.action_space)] * opponent_count
        self.oppdonent_rewards = [0] * opponent_count

        self.go_first = go_first

    def reset(self):
        self.game.reset()

        if not self.go_first:
            print("BoardGameEnvironment: reset - Initialize board with opponent moves")
            self.let_opponents_act()

        self.go_first = not self.go_first

        return self.state

    def step(self, action):
        done = self.game.move(action)

        if not done:
            done = self.let_opponents_act()

        return self.state, self.game.get_reward(), done, {}

    def let_opponents_act(self):
        done = False

        for opponent in self.opponents:
            action = opponent.act(self.state, self.game.get_reward(), done, self.game.legal_actions)

            if action is None:
                break

            done = self.game.move(action)

            if done:
                break

        return done

    def render(self, mode="human"):
        self.game.render()

    @property
    def state(self):
        state = np.zeros(self.observation_space.shape, bool)

        for p in range(self.game.player_count):
            if self.go_first:
                player_id = p + 1
            else:
                player_id = (p + self.game.player_count - 1) % self.game.player_count + 1

            state[p, :, :] = self.game.state == player_id

        return state


class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation, reward, done, legal_actions=None):
        print("Agent: act")
        print(f" - Observation:\n{observation}")
        print(" - Reward from previous action:", reward)

        if done:
            print(" - Not taking action because it's game over!")
            return None

        if legal_actions is not None:
            if len(legal_actions) == 0:
                print(" - Not taking action because there are no legal actions!")
                return None
            action = random.choice(legal_actions)
        else:
            action = self.action_space.sample()

        print(" - Taking Action:", action)

        return action

    def learn(self):
        print("Agent: learn...")


def main():
    env = gym.make("CarRacing-v0")
    game = BoardGame()
    # env = BoardGameEnvironment(game, Agent)
    agent = Agent(env.observation_space, env.action_space)

    # This is our first observation before doing anything
    observation = env.reset()

    # Observation and Action spaces
    print("Observation Space:\n", env.observation_space)
    print("Observation Space Shape:\n", env.observation_space.shape)
    print(f"First Observation:\n{observation}")
    print()

    print("Action Space:\n", env.action_space)
    print(
        "Action Space Length:\n",
        env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape,
    )
    print()

    for _ in range(EPISODE_COUNT):

        observation = env.reset()
        reward = 0
        done = False

        for t in range(MAX_TIMESTEPS):
            if RENDER:
                env.render()

            print(f"\nEpisode Loop: {t+1:04d} Act and Observe")
            action = agent.act(observation, reward, done)

            if done:
                print(f"Episode finished after {t+1} timesteps")
                break

            print("Agent applying action", action)
            observation, reward, done, _ = env.step(action)
            print("Got reward", reward)

        agent.learn()
        print()

    env.close()


if __name__ == "__main__":
    main()
