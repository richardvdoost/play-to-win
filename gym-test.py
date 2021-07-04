# import gym_super_mario_bros
# from agents.human import HumanAgent
from agents.policy_gradient import PolicyGradientAgent
from agents.random import RandomAgent
from brain.activation_functions import LeakyReLU
from environment.board_game import BoardGameEnvironment
from games.tictactoe import TicTacToe

EPOCH_COUNT = 500
EPISODE_COUNT = 500
MAX_TIMESTEPS = 20


def main():
    # env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v0")
    # env = gym.make("CarRacing-v0")
    # env = gym_super_mario_bros.make("SuperMarioBros-v0")

    game = TicTacToe()
    opponents = (RandomAgent(),)
    env = BoardGameEnvironment(game, opponents)

    # agent = RandomAgent(env.observation_space, env.action_space)
    # agent = HumanAgent(game=game)
    agent = PolicyGradientAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_layer_topology=[(64, LeakyReLU)],
        learning_rate=0.0005,
        regularization=1.5,
        discount_rate=0.6,
        experience_batch_size=2 ** 11,
        batch_iterations=512,
        experience_buffer_size=2 ** 16,
        negative_memory_factor=1.1,
        epsilon=0.2,
    )

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

    for epoch in range(EPOCH_COUNT):
        total_reward = 0
        total_episodes = 0
        total_losses = 0

        for _ in range(EPISODE_COUNT):

            observation = env.reset()
            reward = 0
            done = False

            for _ in range(MAX_TIMESTEPS):
                # env.render(mode="console")

                # print(f"\nEpisode Loop: {t+1:04d} Act and Observe")
                action = agent.act(observation, reward, done, legal_actions=game.legal_actions)

                if done:
                    # print(f"Episode finished after {t+1} timesteps")
                    break

                # print("Agent applying action", action)
                observation, reward, done, _ = env.step(action)
                # print("Got reward:", reward)
                # print("Done:", done)

                total_reward += reward
                total_losses += 1 if reward < 0 else 0

            total_episodes += 1

        agent.learn()
        print(
            f"Epoch: {epoch+1:04d} |",
            f"Epsilon: {agent.epsilon:.03f} ({'greedy' if agent.act_greedy else 'non-greedy'}) |",
            f"Reward: {total_reward / total_episodes:.02f} |",
            f"Lost: {total_losses / total_episodes * 100:.01f}% |",
            f"Error: {agent.brain.error:.02f}",
        )

        agent.epsilon *= 0.95
        agent.act_greedy = agent.epsilon < 0.01

    env.close()

    # print(f"Positive:\n{agent.positive_experiences[:3]}")
    # print(f"Negative:\n{agent.negative_experiences[:3]}")


if __name__ == "__main__":
    main()
