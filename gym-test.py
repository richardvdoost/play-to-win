# import gym_super_mario_bros

from agents.human import HumanAgent
from agents.random import RandomAgent
from environment.board_game import BoardGameEnvironment
from games.tictactoe import TicTacToe

EPISODE_COUNT = 20
MAX_TIMESTEPS = 1000
RENDER = True


def main():
    # env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v0")
    # env = gym.make("CarRacing-v0")
    # env = gym_super_mario_bros.make("SuperMarioBros-v0")

    game = TicTacToe()
    env = BoardGameEnvironment(game, RandomAgent)

    # agent = RandomAgent(env.observation_space, env.action_space)
    agent = HumanAgent(game=game)

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
            print("Got reward:", reward)
            print("Done:", done)

        # TODO: agent.learn() or something

        print()

    env.close()


if __name__ == "__main__":
    main()
