import pickle
import time
from pathlib import Path

import numpy as np

from brain import Brain
from brain.activation_functions import Identity
from brain.activation_functions import LeakyReLU
from brain.activation_functions import ReLU
from brain.activation_functions import Sigmoid
from brain.activation_functions import Softmax
from brain.activation_functions import Softplus
from games import TicTacToe
from players import HumanPlayer
from players import PolicyGradientPlayer
from players import RandomPlayer
from plotter import Plotter

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# Hyper parameters
PLAY_COUNT = 1000
EXPERIENCE_BATCH_SIZE = 1024
BATCH_ITERATIONS = 256
EXPERIENCE_BUFFER_SIZE = 2**13
DISCOUNT_RATE = 0.5
LEARNING_RATE = 0.0005
REGULARIZATION = 0.3
BRAIN_TOPOLOGY = (
    (18, None),
    (32, LeakyReLU),
    (32, LeakyReLU),
    (9, Softmax),
)

robot_brain = Brain(
    BRAIN_TOPOLOGY, learning_rate=LEARNING_RATE, regularization=REGULARIZATION
)
learning_robot = PolicyGradientPlayer(
    robot_brain,
    discount_rate=DISCOUNT_RATE,
    batch_iterations=BATCH_ITERATIONS,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)

random_game = TicTacToe((learning_robot, RandomPlayer()))

# Init folders
for path in ("brain/saved", "plots"):
    Path(path).mkdir(exist_ok=True, parents=True)

# Initialize plot data
game_counts = []
wins = []
losses = []
scores = []
mean_experience_values = []
mean_confidences = []
brain_costs = []
brain_costs_ema = []
weight_ranges = []
weight_means = []

# Create a plot figure
plot_data = {
    "score": {
        "placement": 221,
        "graphs": [
            {"label": "Score", "color": "blue"},
            {"label": "% Losses", "color": "red"},
            {"label": "% Wins", "color": "green"},
        ],
        "ylabel": f"Average of {PLAY_COUNT} Games",
        "legend": True,
    },
    "experience": {
        "placement": 222,
        "graphs": [
            {"color": "green", "label": "State Value"},
            {"color": "blue", "label": "Action Confidence"},
        ],
        "ylabel": f"Average Experience",
        "legend": True,
    },
    "cost": {
        "placement": 223,
        "graphs": [{"color": "red_transp"}, {"color": "red"}],
        "ylabel": f"Brain Cost of Last Batch",
        "xlabel": f"Games Played",
    },
    "weights": {
        "placement": 224,
        "graphs": [
            {"color": "blue", "label": "Abs. Max"},
            {"color": "green", "label": "Abs. Mean"},
        ],
        "ylabel": f"Weights Range",
        "xlabel": f"Games Played",
        "legend": True,
    },
}
plotter = Plotter("Policy Network Performance", plot_data)

game_count = 0
prev_score = None
prev_mean_experience_value = None
brain_cost = 0
brain_cost_ema = None
perfect_score_count = 0
start_time = time.time()
running = True
while running:
    try:
        random_game.reset_score()
        random_game.play(PLAY_COUNT)
        learning_robot.learn(BATCH_ITERATIONS)

        game_count += PLAY_COUNT
        score_tuple = random_game.score
        score_tuple_rel = (
            score_tuple[0] / PLAY_COUNT * 100,
            score_tuple[1] / PLAY_COUNT * 100,
        )
        score = score_tuple_rel[0] - score_tuple_rel[1]
        score_diff = (score - prev_score) / abs(prev_score) * 100 if prev_score else 0
        prev_score = score

        brain_cost = robot_brain.cost
        brain_cost_ema = (
            0.9 * brain_cost_ema + 0.1 * brain_cost
            if brain_cost_ema and not np.isnan(brain_cost_ema)
            else brain_cost
        )

        mean_experience_value = learning_robot.mean_experience_value
        mean_experience_value_diff = (
            (mean_experience_value - prev_mean_experience_value)
            / abs(prev_mean_experience_value)
            * 100
            if prev_mean_experience_value
            else 0.0
        )
        prev_mean_experience_value = mean_experience_value
        mean_confidence = learning_robot.confidence

        synapse_stats = robot_brain.synapse_stats

        # Useful info
        print(
            f"Total training time:     {round(time.time() - start_time)} seconds\nGames"
            f" Played:            {game_count}\nScore:                   {score:5.1f}%"
            f" {score_diff:+4.1f}%\nWins / Losses:           {score_tuple_rel[0]:.1f}%"
            f" / {score_tuple_rel[1]:.1f}%\nMean Experience Value:  "
            f" {mean_experience_value:6.3f} {mean_experience_value_diff:+4.1f}%\nMean"
            f" Confidence:         {mean_confidence:6.3f}\nExperience Buffer Usage:"
            f" {learning_robot.experience_buffer_usage * 100:5.1f}%\nBrain Cost:       "
            f"       {brain_cost:4.3f}\nBrain Cost EMA:         "
            f" {(0 if brain_cost_ema is None else brain_cost_ema):4.3f}\nSynapse Stats:"
            f" {synapse_stats}\nOutput: {robot_brain.output[0,:]}\nTarget:"
            f" {robot_brain.target[0,:]}\n"
        )

        # Update plot data
        game_counts.append(game_count)
        wins.append(score_tuple_rel[0])
        losses.append(score_tuple_rel[1])
        scores.append(score)
        mean_experience_values.append(mean_experience_value)
        mean_confidences.append(mean_confidence)
        brain_costs.append(brain_cost)
        brain_costs_ema.append(brain_cost_ema)
        weight_ranges.append(synapse_stats["weight_range"])
        weight_means.append(synapse_stats["weight_mean"])
        graph_data = {
            "score": (scores, losses, wins, game_counts),
            "experience": (mean_experience_values, mean_confidences, game_counts),
            "cost": (brain_costs, brain_costs_ema, game_counts),
            "weights": (weight_ranges, weight_means, game_counts),
        }
        plotter.update_data(graph_data)

        # Stop when we have a perfect score (0 losses)
        if score_tuple[1] == 0:
            print(f"Played perfectly for {PLAY_COUNT} games in a row 😬 Stopping")

            # Save the trained brain, and the plots
            settings_str = (
                f"-batch_iterations={BATCH_ITERATIONS}"
                f"-experience_batch_size={EXPERIENCE_BATCH_SIZE}"
                f"-experience_buffer_size={EXPERIENCE_BUFFER_SIZE}"
                f"-learning_rate={LEARNING_RATE}"
                f"-regularization={REGULARIZATION}"
            )
            for hidden_layer in BRAIN_TOPOLOGY[1:-1]:
                settings_str += (
                    f"hl-neurons={hidden_layer[0]}{hidden_layer[1].__name__}"
                )
            pickle.dump(
                robot_brain, open(f"brain/saved/robot_brain{settings_str}.pickle", "wb")
            )
            plotter.save_image(f"plots/performance-plot{settings_str}.png")

            running = False

    except KeyboardInterrupt:
        running = False

plotter.save_image("plots/performance-plot.png")

experiences_set = (
    [
        experience
        for experience in learning_robot.experiences
        if experience["value"] < 0
    ],
    [
        experience
        for experience in learning_robot.experiences
        if experience["value"] > 0
    ],
)
for experiences in experiences_set:
    for experience in experiences[-10:]:
        np.set_printoptions(precision=5, suppress=True, floatmode="fixed")
        print(f"action probabilities: {experience['action_probabilities']}")
        print(
            f"value: {experience['value']:6.3f}            "
            f" {experience['choice'] * '        '}<{experience['choice']}>"
        )

        np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
        print(f"nudge (k):            {experience['nudge'] / 1e3}")
        print()
