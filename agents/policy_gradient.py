import numpy as np
from brain.activation_functions import Softmax
from brain.brain import Brain

from agents.agent import Agent


class PolicyGradientAgent(Agent):
    def __init__(
        self,
        observation_space=None,
        action_space=None,
        game=None,
        hidden_layer_topology=[],
        output_layer_type=Softmax,
        learning_rate=0.001,
        regularization=0.5,
        discount_rate=0.5,
        experience_batch_size=2 ** 10,
        batch_iterations=32,
        experience_buffer_size=2 ** 15,
        negative_memory_factor=1.0,
        epsilon=None,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.game = game

        self.hidden_layer_topology = hidden_layer_topology
        self.output_layer_type = output_layer_type
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.create_brain()

        self.discount_rate = discount_rate
        self.experience_batch_size = experience_batch_size
        self.batch_iterations = batch_iterations
        self.experience_buffer_size = experience_buffer_size
        self.negative_memory_factor = negative_memory_factor
        self.epsilon = epsilon

        self.episode = []
        self.positive_experiences = []
        self.negative_experiences = []
        self.act_greedy = False
        self.learn_while_playing = False

    def create_brain(self):
        brain_topology = (
            [(self.observation_space_length, None)]
            + self.hidden_layer_topology
            + [(self.action_space_length, self.output_layer_type)]
        )
        self.brain = Brain(
            brain_topology,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
        )

    def act(self, observation, reward, done, legal_actions=None):
        observation_space_shape = (1, self.observation_space_length)
        action_space_shape = (1, self.action_space_length)

        legal_actions_mask = np.zeros(action_space_shape, bool)

        if done:
            self.game_over(reward)
            return None

        else:
            reshaped_observation = observation.reshape(observation_space_shape)

            if legal_actions is None:
                legal_actions_mask += True
            else:
                legal_actions_mask[0, legal_actions] = True

            action_probabilities = self.brain.think(reshaped_observation).copy()

            # if self.show_action_probabilities:
            #     ticks = self.show_action_probabilities * 60
            #     for tick in range(int(ticks)):
            #         scale = 1 - (1 - (tick / ticks)) ** 2
            #         game.render(action_probabilities=action_probabilities * scale)
            #         game.clock.tick(60)

            if self.epsilon is not None and np.random.rand() < self.epsilon:
                action = np.random.choice(legal_actions)

            elif self.act_greedy:
                action = (action_probabilities * legal_actions_mask + 1e-8 * legal_actions_mask).argmax()

            else:
                # Sample an action over the softmax probabilities
                probabilities = action_probabilities

                if legal_actions is None:
                    action = np.random.choice(self.action_space_length, p=probabilities.flatten())
                else:
                    probabilities *= legal_actions_mask

                    try:
                        for _ in range(3):
                            p_sum = probabilities.sum()
                            if p_sum > 1 - 1e-9:
                                break
                            probabilities /= p_sum
                        action = np.random.choice(self.action_space_length, p=probabilities.flatten())

                    except Exception as e:
                        print("probabilities:", probabilities)
                        print("probabilities sum:", probabilities.sum())
                        action = np.random.choice(legal_actions)
                        raise e

        self.episode.append(
            {
                "observation": reshaped_observation,
                "legal_actions_mask": legal_actions_mask,
                "action_probabilities": action_probabilities,
                "action": action,
                "confidence": action_probabilities[0, action],
                "reward": reward,
            }
        )

        return action

    def reward(self, reward):
        self.episode[-1]["reward"] = reward

    def game_over(self, reward):
        self.episode.append(
            {
                "reward": reward,
                "confidence": 1.0,
                "legal_actions_mask": np.zeros((1, self.action_space_length), bool),
            }
        )

        self.process_last_experiences()

        if self.learn_while_playing:
            self.learn()

    def learn(self):
        if len(self.positive_experiences) + len(self.negative_experiences) == 0:
            return

        for batch in self.get_experience_batches():
            batch_length = len(batch)
            shape = (batch_length, -1)

            inputs = np.array([exp["observation"] for exp in batch]).reshape(shape)
            nudges = np.array([exp["nudge"] for exp in batch]).reshape(shape)

            self.brain.nudge(inputs, nudges)

    def process_last_experiences(self):
        """
        Go over all experiences in the last episode in reverse order. Assign values to every action taken
        based on the reward gathered and the value of the next state (discounted with how confident we are
        to arrive in that next state)
        """

        experience_value = 0
        for experience in reversed(self.episode):
            experience_value = (
                experience["reward"] + experience_value * self.discount_rate * experience["confidence"]
            )
            experience["value"] = experience_value

            # Skip this experience if we did not have to choose (still count the value above)
            allowed_actions_length = np.count_nonzero(experience["legal_actions_mask"])
            if allowed_actions_length < 2:
                continue

            # Only remember the experience if it had a significant value (negative or positive)
            if -1e-4 < experience_value < 1e-4:
                continue

            experience["nudge"] = np.zeros(experience["legal_actions_mask"].shape)
            illegal_actions_mask = experience["legal_actions_mask"] == False
            experience["nudge"][illegal_actions_mask] -= (
                1e3 * experience["action_probabilities"][illegal_actions_mask]
            )

            # Separate positive and negative experiences
            half_buffer_size = self.experience_buffer_size // 2
            if experience_value > 0:
                nudge = experience_value
                experiences = self.positive_experiences
            else:
                # Nudge harder on negative experiences
                # (try to avoid getting overconfident when winning the majority of the time)
                nudge = self.negative_memory_factor * experience_value
                experiences = self.negative_experiences

            experience["nudge"][0, experience["action"]] = nudge

            # If we can add another experience do so, otherwise, replace a random experience
            if len(experiences) < half_buffer_size:
                experiences.append(experience)
            else:
                random_experience_index = np.random.choice(half_buffer_size)
                experiences[random_experience_index] = experience

        self.episode = []

    def get_experience_batches(self):
        """
        Return a number of random sample batches of past experiences
        (for batch gradient descent of the brain)

        Keep a 50/50 balance between positive and negative experiences
        """
        batches = []

        positive_experiences_length = len(self.positive_experiences)
        negative_experiences_length = len(self.negative_experiences)

        if positive_experiences_length == 0 or negative_experiences_length == 0:
            return batches

        half_batch_length = self.experience_batch_size // 2

        for _ in range(self.batch_iterations):
            positive_experience_indexes = np.random.choice(
                positive_experiences_length,
                half_batch_length,
                replace=half_batch_length > positive_experiences_length,
            )
            negative_experience_indexes = np.random.choice(
                negative_experiences_length,
                half_batch_length,
                replace=half_batch_length > negative_experiences_length,
            )
            batches.append(
                [self.positive_experiences[i] for i in positive_experience_indexes]
                + [self.negative_experiences[i] for i in negative_experience_indexes]
            )

        return batches

    # def pick_action(self, action_probabilities):
    #     """
    #     From a set of probabilities (of actions), pick one based on the probability and return the index.
    #     """
    #     prob_sum = np.sum(action_probabilities)
    #     action_probabilities /= prob_sum
    #     return np.random.choice(action_probabilities.size, p=action_probabilities.flatten())

    # def create_action_matrix(self, action_space_length, choice):
    #     """
    #     Create a boolean action matrix using one-hot encoding (all false except the choice).
    #     """
    #     action = np.zeros((1, action_space_length), dtype=bool)
    #     action[0, choice] = True
    #     return action

    # @staticmethod
    # def softmax(X):
    #     exp_X = np.exp(X)
    #     return exp_X / np.sum(exp_X)

    @property
    def experience_buffer_usage(self):
        return (
            len(self.positive_experiences) + len(self.negative_experiences)
        ) / self.experience_buffer_size

    @property
    def mean_experience_value(self):
        experience_values = np.array(
            [experience["value"] for experience in self.positive_experiences + self.negative_experiences]
        )
        return experience_values.mean()

    @property
    def confidence(self):
        confidence_values = np.array(
            [
                experience["confidence"]
                for experience in self.positive_experiences + self.negative_experiences
            ]
        )
        return confidence_values.mean()
