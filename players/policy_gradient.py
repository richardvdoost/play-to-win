import numpy as np

from brain import Brain
from .player import Player


class PolicyGradientPlayer(Player):
    def __init__(
        self,
        brain,
        discount_factor=0.5,
        reward_factor=0.5,
        experience_batch_size=128,
        batch_iterations=4,
        experience_buffer_size=2048,
    ):

        self.brain = brain
        self.discount_factor = discount_factor
        self.reward_factor = reward_factor
        self.experience_batch_size = experience_batch_size
        self.batch_iterations = batch_iterations
        self.experience_buffer_size = experience_buffer_size

        self.episode = []
        self.experiences = []
        self.is_learning = True
        self.act_greedy = False

    def take_action(self, state, allowed_actions):

        action_count = allowed_actions.size

        player_count = 2
        player_states = []
        for player_index in range(player_count):
            player_states.append(state == player_index)

        state_reshaped = np.concatenate(player_states) - 0.5
        state_reshaped.shape = 1, state_reshaped.size
        allowed_actions_reshaped = allowed_actions.reshape(1, action_count)

        # Feed the state into the brain to get the probablity distribution of actions
        action_probabilities = self.brain.think(state_reshaped).copy()

        if self.act_greedy:
            choice = (action_probabilities * allowed_actions_reshaped + 1e-8 * allowed_actions_reshaped).argmax()

        else:
            # Sample an action over the softmax probabilities
            for _ in range(32):
                choice = np.random.choice(action_probabilities.size, p=action_probabilities.flatten())
                if allowed_actions_reshaped[0, choice]:
                    break

            if not allowed_actions_reshaped[0, choice]:
                choice = (action_probabilities * allowed_actions_reshaped + 1e-8 * allowed_actions_reshaped).argmax()

        self.episode.append(
            {
                "state": state_reshaped,
                "allowed_actions": allowed_actions_reshaped,
                "action_probabilities": action_probabilities,
                "choice": choice,
                "value": 0,
            }
        )

        action = self.create_action_matrix(action_count, choice)

        return action.reshape(allowed_actions.shape)

    def reward(self, reward):
        if self.is_learning:
            self.episode[-1]["value"] += reward

    def game_over(self):

        self.process_last_experiences()

        if not self.is_learning:
            return

        if len(self.experiences) < self.experience_batch_size:
            return

        for batch in self.get_experience_batches():

            samples = []
            for experience in batch:
                samples.append({"input": experience["state"].flatten(), "nudge": experience["nudge"].flatten()})

            self.brain.nudge(samples)

    def process_last_experiences(self):
        """
        Go over all experiences in the last episode in reverse order. Assign values to every action taken based on the
        reward gathered (already stored in 'value') and decay with some discount factor. Store the valued experiences.
        """

        next_experience_value = 0
        for experience in reversed(self.episode):

            experience["value"] += self.discount_factor * next_experience_value
            next_experience_value = experience["value"]

            # Skip this experience if we did not have to choose (still count the value above)
            allowed_actions_count = np.count_nonzero(experience["allowed_actions"])
            if allowed_actions_count < 2:
                continue

            # Only remember the experience if it had a significant value (negative or positive)
            if -1e-4 < experience["value"] < 1e-4:
                continue

            # # Hand tuned way to recreate target
            # target = np.log(0.001 + experience["action_probabilities"] * 0.998)
            # target[0, experience["choice"]] += (
            #     experience["value"] * 1.2 / (1.2 - target[0, experience["choice"]]) * self.reward_factor
            # )
            # target[experience["allowed_actions"] == False] -= 1e4
            # t = np.exp(target)
            # target = t / np.sum(t)

            # # Use the alpha-toe method but indirectly by creating a target rather than using the gradient
            # choice = experience["choice"]
            # target = experience["action_probabilities"] * experience["allowed_actions"]
            # target[0, choice] += max(1e-2, target[0, choice]) * experience["value"]

            # experience["target"] = target

            # NEW WAY: Don't use a 'target' but use a 'nudge' to nudge the brain into a certain direction regardless of
            # the current output the brain gave at the time

            experience["nudge"] = np.zeros(experience["allowed_actions"].shape)
            experience["nudge"][experience["allowed_actions"] == False] -= 1e4
            experience["nudge"][0, experience["choice"]] = experience["value"] * self.reward_factor

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
        batches = []
        for _ in range(self.batch_iterations):
            experience_indexes = np.random.choice(
                experience_count, self.experience_batch_size, replace=self.experience_batch_size > experience_count
            )
            batches.append([self.experiences[i] for i in experience_indexes])

        return batches

    def pick_action(self, action_probabilities):
        """
        From a set of probabilities (of actions), pick one based on the probability and return the index.
        """
        prob_sum = np.sum(action_probabilities)
        action_probabilities /= prob_sum
        return np.random.choice(action_probabilities.size, p=action_probabilities.flatten())

    def create_action_matrix(self, action_count, choice):
        """
        Create a boolean action matrix using one-hot encoding (all false except the choice).
        """
        action = np.zeros((1, action_count), dtype=bool)
        action[0, choice] = True
        return action

    @staticmethod
    def softmax(X):
        exp_X = np.exp(X)
        return exp_X / np.sum(exp_X)

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
