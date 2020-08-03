import numpy as np

from .player import Player


class PolicyGradientPlayer(Player):
    def __init__(
        self,
        brain,
        discount_factor=0.5,
        reward_factor=0.5,
        experience_batch_size=16,
        batch_iterations=1,
        experience_buffer_size=1024,
        epsilon=None,
    ):

        self.brain = brain
        self.discount_factor = discount_factor
        self.reward_factor = reward_factor
        self.experience_batch_size = experience_batch_size
        self.batch_iterations = batch_iterations
        self.experience_buffer_size = experience_buffer_size
        self.epsilon = epsilon

        self.episode = []
        self.experiences = []
        self.act_greedy = False
        self.learn_while_playing = False

    def take_action(self, game):

        if self.learn_while_playing:
            self.learn()

        state = game.state
        allowed_actions = game.allowed_actions

        action_count = allowed_actions.size

        player_states = [state == (self.index + offset) % game.player_count for offset in range(game.player_count)]
        state_reshaped = np.concatenate(player_states) - 0.5
        state_reshaped.shape = 1, state_reshaped.size
        allowed_actions_reshaped = allowed_actions.reshape(1, action_count)

        # Feed the state into the brain to get the probablity distribution of actions
        action_probabilities = self.brain.think(state_reshaped).copy()

        if self.epsilon is not None and np.random.rand() < self.epsilon:
            allowed_coords = np.argwhere(allowed_actions)
            allowed_coord_count = len(allowed_coords)
            choice = np.random.choice(allowed_coord_count)

        elif self.act_greedy:
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
                # "action_probabilities": action_probabilities,
                "choice": choice,
                "reward": 0,
            }
        )

        action = self.create_action_matrix(action_count, choice)

        return action.reshape(allowed_actions.shape)

    def reward(self, reward):
        self.episode[-1]["reward"] = reward

    def game_over(self, _game):
        self.process_last_experiences()

    def learn(self, batch_iterations=None):
        if len(self.experiences) < 1:
            return

        for batch in self.get_experience_batches(count=batch_iterations):
            samples = []
            for experience in batch:
                samples.append({"input": experience["state"].flatten(), "nudge": experience["nudge"].flatten()})

            self.brain.nudge(samples)

    def process_last_experiences(self):
        """
        Go over all experiences in the last episode in reverse order. Assign values to every action taken based on the
        reward gathered (already stored in 'value') and decay with some discount factor. Store the valued experiences.
        """

        experience_value = 0
        for experience in reversed(self.episode):
            experience_value = experience["reward"] + self.discount_factor * experience_value

            # Skip this experience if we did not have to choose (still count the value above)
            allowed_actions_count = np.count_nonzero(experience["allowed_actions"])
            if allowed_actions_count < 2:
                continue

            # Only remember the experience if it had a significant value (negative or positive)
            if -1e-4 < experience_value < 1e-4:
                continue

            experience["nudge"] = np.zeros(experience["allowed_actions"].shape)
            experience["nudge"][experience["allowed_actions"] == False] -= 1e3
            experience["nudge"][0, experience["choice"]] = experience_value * self.reward_factor

            # If we can add another experience do so, otherwise, replace a random experience
            if len(self.experiences) < self.experience_buffer_size:
                self.experiences.append(experience)
            else:
                random_experience_index = np.random.choice(self.experience_buffer_size)
                self.experiences[random_experience_index] = experience

        self.episode = []

    def get_experience_batches(self, count=None):
        """
        Return a number of random sample batches of past experiences (for batch gradient descent of the brain)
        """
        experience_count = len(self.experiences)
        batches = []
        for _ in range(self.batch_iterations if count is None else count):
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
        experience_values = np.array([experience["reward"] for experience in self.experiences])
        return experience_values.mean()
