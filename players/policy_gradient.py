import numpy as np

from .player import Player


class PolicyGradientPlayer(Player):
    def __init__(
        self,
        brain,
        discount_rate=0.5,
        experience_batch_size=16,
        batch_iterations=1,
        experience_buffer_size=1024,
        epsilon=None,
        negative_memory_factor=1,
    ):

        self.brain = brain
        self.discount_rate = discount_rate
        self.experience_batch_size = experience_batch_size
        self.batch_iterations = batch_iterations
        self.experience_buffer_size = experience_buffer_size
        self.epsilon = epsilon
        self.negative_memory_factor = negative_memory_factor

        self.episode = []
        self.positive_experiences = []
        self.negative_experiences = []
        self.act_greedy = False
        self.learn_while_playing = False

    def take_action(self, game):

        state = game.state
        allowed_actions = game.allowed_actions

        action_count = allowed_actions.size

        player_states = [
            state == (self.index + offset) % game.player_count for offset in range(game.player_count)
        ]
        state_reshaped = np.concatenate(player_states) - 0.5
        state_reshaped.shape = 1, state_reshaped.size
        allowed_actions_reshaped = allowed_actions.reshape(1, action_count)

        # Feed the state into the brain to get the probablity distribution of actions
        action_probabilities = self.brain.think(state_reshaped).copy()

        if self.show_action_probabilities:
            ticks = self.show_action_probabilities * 60
            for tick in range(int(ticks)):
                scale = 1 - (1 - (tick / ticks)) ** 2
                game.render(action_probabilities=action_probabilities * scale)
                game.clock.tick(60)

        if self.epsilon is not None and np.random.rand() < self.epsilon:
            allowed_coords = np.argwhere(allowed_actions_reshaped)
            allowed_coord_count = len(allowed_coords)
            choice = allowed_coords[np.random.choice(allowed_coord_count)][1]

        elif self.act_greedy:
            choice = (
                action_probabilities * allowed_actions_reshaped + 1e-8 * allowed_actions_reshaped
            ).argmax()

        else:
            # Sample an action over the softmax probabilities
            for _ in range(16):
                choice = np.random.choice(action_probabilities.size, p=action_probabilities.flatten())
                if allowed_actions_reshaped[0, choice]:
                    break

            if not allowed_actions_reshaped[0, choice]:
                choice = (
                    action_probabilities * allowed_actions_reshaped + 1e-8 * allowed_actions_reshaped
                ).argmax()

        # Remember how confident we are to take the action we're about to take
        confidence = action_probabilities[0, choice]

        self.episode.append(
            {
                "state": state_reshaped,
                "allowed_actions": allowed_actions_reshaped,
                "action_probabilities": action_probabilities,  # Debug only
                "choice": choice,
                "confidence": confidence,
                "reward": 0,
            }
        )

        action = self.create_action_matrix(action_count, choice)

        return action.reshape(allowed_actions.shape)

    def reward(self, reward):
        self.episode[-1]["reward"] = reward

    def game_over(self, _game):
        self.process_last_experiences()

        if self.learn_while_playing:
            self.learn()

    def learn(self):
        if len(self.positive_experiences) + len(self.negative_experiences) == 0:
            return

        for batch in self.get_experience_batches():
            samples = []
            for experience in batch:
                samples.append(
                    {"input": experience["state"].flatten(), "nudge": experience["nudge"].flatten()}
                )

            self.brain.nudge(samples)

    def process_last_experiences(self):
        """
        Go over all experiences in the last episode in reverse order. Assign values to every action taken based on the
        reward gathered and the value of the next state (discounted with how confident we are to arrive in that next state)
        """

        experience_value = 0
        for experience in reversed(self.episode):
            experience_value = (
                experience["reward"] + experience_value * self.discount_rate * experience["confidence"]
            )
            experience["value"] = experience_value

            # Skip this experience if we did not have to choose (still count the value above)
            allowed_actions_count = np.count_nonzero(experience["allowed_actions"])
            if allowed_actions_count < 2:
                continue

            # Only remember the experience if it had a significant value (negative or positive)
            if -1e-4 < experience_value < 1e-4:
                continue

            experience["nudge"] = np.zeros(experience["allowed_actions"].shape)
            forbidden_mask = experience["allowed_actions"] == False
            experience["nudge"][forbidden_mask] -= (
                1e3 * experience["action_probabilities"][forbidden_mask]
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

            experience["nudge"][0, experience["choice"]] = nudge

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
        positive_experience_count = len(self.positive_experiences)
        negative_experience_count = len(self.negative_experiences)

        half_batch = self.experience_batch_size // 2

        batches = []
        for _ in range(self.batch_iterations):
            positive_experience_indexes = np.random.choice(
                positive_experience_count,
                half_batch,
                replace=half_batch > positive_experience_count,
            )
            negative_experience_indexes = np.random.choice(
                negative_experience_count,
                half_batch,
                replace=half_batch > negative_experience_count,
            )
            batches.append(
                [self.positive_experiences[i] for i in positive_experience_indexes]
                + [self.negative_experiences[i] for i in negative_experience_indexes]
            )

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
