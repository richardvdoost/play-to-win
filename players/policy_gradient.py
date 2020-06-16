import numpy as np
from brain import Brain
from .player import Player


class PolicyGradientPlayer(Player):
    def __init__(
        self,
        brain,
        discount_factor=0.5,
        reward_factor=0.5,
        experience_batch_size=64,
        train_iterations=1,
        experience_buffer_size=1024,
    ):

        self.brain = brain
        self.discount_factor = discount_factor
        self.reward_factor = reward_factor
        self.experience_batch_size = experience_batch_size
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

        exploded_state = np.concatenate(player_states) - 0.5
        exploded_state.shape = 1, exploded_state.size

        allowed_actions_1d = allowed_actions.reshape(1, action_count)

        # Feed the state into the brain to get the probablity distribution of actions
        action_probabilities = self.brain.think(exploded_state)
        action_probabilities[allowed_actions_1d == False] = 0
        action_probabilities[allowed_actions_1d] += 1e-8  # Prevent division by 0
        choice = self.pick_action(action_probabilities)
        action = self.create_action_matrix(action_count, choice)

        self.episode.append(
            {"state": exploded_state, "action_probabilities": action_probabilities, "choice": choice, "value": 0}
        )

        return action.reshape(allowed_actions.shape)

    def reward(self, reward):
        self.episode[-1]["value"] += reward

    def game_over(self):

        if not self.is_learning:
            return

        self.process_last_experiences()

        if len(self.experiences) < self.experience_batch_size:
            return

        for batch in self.get_experience_batches():

            training_data = []
            for experience in batch:
                training_data.append({"input": experience["state"].flatten(), "target": experience["target"].flatten()})

            self.brain.train(training_data, self.train_iterations)

    def process_last_experiences(self):
        """
        Go over all experiences in the last episode in reverse order. Assign values to every action taken based on the
        reward gathered (already stored in 'value') and decay with some discount factor. Store the valued experiences.
        """

        next_experience_value = 0
        for experience in reversed(self.episode):

            experience["value"] += self.discount_factor * next_experience_value
            next_experience_value = experience["value"]

            # Only remember the experience if it had a significant value (negative or positive)
            if -1e-4 < experience["value"] < 1e-4:
                continue

            # Try to figure out what to do when we have a similar experience next time
            # - Positive value: Increase the probability of our previous choice
            # - Negative value: Decrease the probability of our previous choice
            updated_action_probabilities = experience["action_probabilities"].copy()
            updated_action_probabilities[0, experience["choice"]] *= np.exp(self.reward_factor * experience["value"])
            updated_action_probabilities /= sum(updated_action_probabilities[0, :])
            experience["target"] = updated_action_probabilities.flatten()

            # If we can add another experience do so, otherwise, replace a random experience
            if len(self.experiences) < self.experience_buffer_size:
                self.experiences.append(experience)
            else:
                random_experience_index = np.random.choice(self.experience_buffer_size)
                self.experiences[random_experience_index] = experience

        self.episode = []

    def get_experience_batches(self):
        """
        Return a number of random sample batches of past experiences (for batch gradient descent of the brain)
        """
        experience_count = len(self.experiences)
        experience_indexes = np.random.choice(experience_count, experience_count, replace=False)

        batch_size = min(self.experience_batch_size, experience_count)
        batches = []
        for i in range(0, experience_count, batch_size):
            until = min(i + batch_size, experience_count)
            experience_indexes_batch = experience_indexes[i:until]
            batches.append([self.experiences[i] for i in experience_indexes_batch])

        return batches

    def pick_action(self, action_probabilities):
        """
        From a set of probabilities (of actions), pick one based on the probability and return the index.
        """
        prob_sum = sum(action_probabilities[0, :])
        action_probabilities /= prob_sum
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
