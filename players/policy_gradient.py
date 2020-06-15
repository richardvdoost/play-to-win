import numpy as np
from brain import Brain
from .player import Player


class PolicyGradientPlayer(Player):
    def __init__(
        self,
        brain,
        discount_factor=0.5,
        reward_scale=0.5,
        mini_batch_size=100,
        train_iterations=1,
        experience_buffer_size=1000,
    ):

        self.brain = brain
        self.discount_factor = discount_factor
        self.reward_scale = reward_scale
        self.mini_batch_size = mini_batch_size
        self.train_iterations = train_iterations
        self.experience_buffer_size = experience_buffer_size

        self.episode = []
        self.experiences = []
        self.is_learning = True

    def take_action(self, state, allowed_actions):

        action_count = allowed_actions.size

        player_count = 2
        player_states = []
        for player_index in range(player_count):
            player_states.append(state == player_index)

        # print(f"state: {state}")
        exploded_state = np.concatenate(player_states) - 0.5  # Make the range [-0.5, 0.5]
        # print(f"exploded state: {exploded_state}")
        exploded_state.shape = 1, exploded_state.size
        # print(f"flat exploded state: {exploded_state}")

        allowed_actions_1d = allowed_actions.reshape(1, action_count)

        # Feed the state into the brain to get the probablity distribution of actions
        action_probabilities = self.brain.think(exploded_state)
        action_probabilities[allowed_actions_1d == False] = 0
        choice = self.pick_action(action_probabilities)
        # print(f"action probablilities: {action_probabilities} - chosen: {choice}")

        # Create a new action matrix
        action = self.create_action_matrix(action_count, choice)

        self.episode.append(
            {"state": exploded_state, "action_probabilities": action_probabilities, "choice": choice, "value": 0,}
        )

        return action.reshape(allowed_actions.shape)

    def reward(self, reward):
        self.episode[-1]["value"] += reward

    def game_over(self):

        if not self.is_learning:
            return

        self.value_episode_experiences()

        action_count = self.experiences[0]["action_probabilities"].shape[1]

        training_data = []
        for experience in self.get_experience_sample():

            # Create X and Y sample for the brain
            input_values = experience["state"].flatten()

            # Modify the action probability of the action we took, and take a new sample to determine the label Y for
            # the current state X. For positive rewards, we want to increase the probability of the action we took, for
            # negative rewards, we want to decrease the probability.

            # Multiplying by exp(value) will have the desired behaviour

            updated_action_probabilities = experience["action_probabilities"].copy()
            updated_action_probabilities[0, experience["choice"]] *= np.exp(self.reward_scale * experience["value"])

            # After changing the action probabilities using the experienced value/reward, reconsider the action choice
            reconsidered_choice = self.pick_action(updated_action_probabilities)
            target_values = self.create_action_matrix(action_count, reconsidered_choice).flatten()

            training_data.append({"input": input_values, "target": target_values})

        self.brain.train(training_data, self.train_iterations)

    def value_episode_experiences(self):
        """
        Go over all experiences in the last episode in reverse order. Assign values to every action taken based on the
        reward gathered (already stored in 'value') and decay with some discount factor. Store the valued experiences.
        """
        last_reward = 0
        for experience in reversed(self.episode):
            experience["value"] += self.discount_factor * last_reward
            last_reward = experience["value"]

            # If we can add another experience do so, otherwise, replace a random experience
            if len(self.experiences) < self.experience_buffer_size:
                self.experiences.append(experience)
            else:
                random_experience_index = np.random.choice(self.experience_buffer_size)
                self.experiences[random_experience_index] = experience

        self.episode = []

    def get_experience_sample(self):
        """
        Return a random sample of past experiences
        """
        experience_count = len(self.experiences)
        sample_count = min(self.mini_batch_size, experience_count)
        experience_indexes = np.random.choice(experience_count, sample_count)
        return [self.experiences[i] for i in experience_indexes]

    def pick_action(self, action_probabilities):
        """
        From a set of probabilities (of actions), pick one based on the probability and return the index.
        """
        action_probabilities /= sum(action_probabilities[0, :])
        return np.random.choice(action_probabilities.size, p=action_probabilities[0, :])

    def create_action_matrix(self, action_count, choice):
        """
        Create a boolean action matrix using one-hot encoding (all false except the choice).
        """
        action = np.zeros((1, action_count), dtype=bool)
        action[0, choice] = True
        return action

    @property
    def experience_buffer_usage(self):
        return len(self.experiences) / self.experience_buffer_size

    @property
    def mean_experience_value(self):
        experience_values = np.array([experience["value"] for experience in self.experiences])
        return experience_values.mean()

    def __str__(self):
        return (
            f"Brain cost: {self.brain.cost():4.3f} - Mean experience value: {self.mean_experience_value:6.3f}\n\n"
            f"Output: {self.brain.output[1,:]}\n"
            f"Target: {self.brain.target[1,:]}"
        )
