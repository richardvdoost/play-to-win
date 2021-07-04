import random
import time

from agents.agent import Agent


class RandomAgent(Agent):
    debug = False

    def act(self, observation, reward, done, legal_actions=None):
        if self.debug:
            print("Agent: act")
            print(f" - Observation:\n{observation}")
            print(" - Reward from previous action:", reward)
            time.sleep(1)

        if done:
            if self.debug:
                print(" - Not taking action because it's game over!")
            return None

        if legal_actions is not None:
            if len(legal_actions) == 0:
                print("ERROR: Not taking action because there are no legal actions!")
                return None
            action = random.choice(legal_actions)
        else:
            action = self.action_space.sample()

        if self.debug:
            print(" - Taking Action:", action)

        return action
