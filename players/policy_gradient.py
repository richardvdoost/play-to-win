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
        action_probabilities += 1e-8  # Prevent division by 0
        action_probabilities[allowed_actions_reshaped == False] = 0
        action_probabilities /= np.sum(action_probabilities)  # Normalize
        choice = np.random.choice(action_probabilities.size, p=action_probabilities.flatten())

        action = self.create_action_matrix(action_count, choice)

        self.episode.append(
            {
                "state": state_reshaped,
                "allowed_actions": allowed_actions_reshaped,
                "action_probabilities": action_probabilities,
                "choice": choice,
                "value": 0,
            }
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

        batch_data = []
        for batch in self.get_experience_batches():

            training_data = []
            for experience in batch:
                training_data.append({"input": experience["state"].flatten(), "target": experience["target"].flatten()})

            batch_data.append(training_data)

        # Cycle through the mini batches and train the brain on them (multiple cycles for more experience efficiency)
        for _ in range(self.batch_iterations):
            for training_data in batch_data:
                self.brain.train(training_data, 1)

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

            # Try to figure out what to do when we have a similar experience next time

            # Convert action probabilities to expected value (inverse softmax = ln(x) + c)
            # Now, add the value of the experience to the expected value of the chosen action
            # Convert back to probability distribution with softmax
            # print(f"action probabilities:     {experience['action_probabilities']}")
            # print(f"choice: {experience['choice']} - value: {experience['value']}")

            # Attempt 3:
            # Nudge the probability of the chosen action in the experience
            #  - Use the inverse sigmoid to compute the current "z" value of the action probability
            #  - Compute the new "z" value simply by adding the value of the experience to it
            #  - Compute the new probability with sigmoid(z)
            #  - Scale the other probabilities so that the total sum of probabilities is 1

            # If there's only one allowed action, don't bother updating probabilities
            target = experience["action_probabilities"].copy()
            if allowed_actions_count > 1:

                # print(f"Value: {experience['value']}")
                # print(f"Old action probabilities: {target}")
                # p = target[0, experience["choice"]]
                # print(f"Chosen action {experience['choice']} probabilty before: {p}")
                # z = np.log(p / (1 - p))
                # print(f"Z Value: {z}")
                # z += self.reward_factor * experience["value"]
                # print(f"New Z Value: {z}")
                # p = 1 / (1 + np.exp(-z))
                # print(f"Chosen action probabilty after:  {p}")
                # target[0, experience["choice"]] = p

                # other_actions_mask = experience["allowed_actions"].copy()
                # other_actions_mask[0, experience["choice"]] = False
                # target[other_actions_mask] *= (1 - p) / np.sum(target[other_actions_mask])

                other_actions_mask = experience["allowed_actions"].copy()
                other_actions_mask[0, experience["choice"]] = False
                other_actions_sum = np.sum(target[other_actions_mask])

                # Attempt 4:
                # print(f"Old action probabilities: {target}")
                # print(f"Value: {experience['value']}")
                p = target[0, experience["choice"]]
                # print(f"Chosen action {experience['choice']} probabilty before: {p}")
                if experience["value"] > 0:
                    p = 1 - (1 - p) * np.exp(-experience["value"] * self.reward_factor)

                    # Decrease other probabilities
                    target[other_actions_mask] *= (1 - p) / other_actions_sum
                else:
                    p = p * np.exp(experience["value"] * self.reward_factor)

                    # Increase other probabilities
                    target[other_actions_mask] += (1 - p - other_actions_sum) / np.count_nonzero(other_actions_mask)

                # print(f"Chosen action probabilty after:  {p}")

                target[0, experience["choice"]] = p

                # print(f"New action probabilities: {target}")
                # print()

                # if len(self.experiences) > 5:
                #     exit()

            experience["target"] = target

            # target = np.ma.log(experience["action_probabilities"])
            # target = experience["action_probabilities"]
            # target[experience["allowed_actions"] == False] = 0
            # print(f"target before:            {target}")
            # target[0, experience["choice"]] += self.reward_factor * experience["value"]
            # print(f"target after:             {target}")
            # target[experience["allowed_actions"]] = self.softmax(target[experience["allowed_actions"]])
            # experience["target"] = target
            # print(f"new action probabilities: {target}")
            # print(f"allowed actions:          {experience['allowed_actions']}")

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
