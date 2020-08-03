import pickle

import numpy as np

from brain import Brain
from brain.activation_functions import Identity, ReLU, Sigmoid, Softmax, Softplus
from games import ConnectFour
from players import PolicyGradientPlayer, RandomPlayer, HumanPlayer
from plotter import Plotter

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

RANDOM_GAME_COUNT = 1000
TRAIN_GAME_COUNT = 1000
DISCOUNT_FACTOR = 0.95
REWARD_FACTOR = 2
EXPERIENCE_BATCH_SIZE = 2048
BATCH_ITERATIONS = 64
EXPERIENCE_BUFFER_SIZE = 2 ** 20

LEARNING_RATE = 0.0001
REGULARIZATION = 0.5

BRAIN_TOPOLOGY = (
    (84, None),
    (256, ReLU),
    (256, Sigmoid),
    (256, ReLU),
    (256, Sigmoid),
    (256, ReLU),
    (7, Softmax),
)
BRAIN_FILEPATH = "brain/saved/connect-four-brain.pickle"

try:
    robot_brain = pickle.load(open(BRAIN_FILEPATH, "rb"))
except Exception:
    robot_brain = Brain(BRAIN_TOPOLOGY, learning_rate=LEARNING_RATE, regularization=REGULARIZATION)

# The robots
learner_robot = PolicyGradientPlayer(
    robot_brain,
    discount_factor=DISCOUNT_FACTOR,
    reward_factor=REWARD_FACTOR,
    batch_iterations=BATCH_ITERATIONS,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)

trainer_robot = PolicyGradientPlayer(
    robot_brain,
    discount_factor=DISCOUNT_FACTOR,
    reward_factor=REWARD_FACTOR,
    batch_iterations=BATCH_ITERATIONS,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)

# The games
train_game = ConnectFour((learner_robot, trainer_robot))
random_game = ConnectFour((learner_robot, RandomPlayer()))

# Initialize plot data
game_counts = []
wins = []
losses = []
scores = []
mean_experience_values = []
brain_costs = []
brain_costs_ema = []
weight_ranges = []

# Create a plot figure
plot_data = {
    "score": {
        "placement": 221,
        "graphs": [
            {"label": "Score", "color": "blue"},
            {"label": "% Losses", "color": "red"},
            {"label": "% Wins", "color": "green"},
        ],
        "ylabel": f"Average of {RANDOM_GAME_COUNT} Games",
        "legend": True,
    },
    "value": {"placement": 222, "graphs": [{"color": "blue"}], "ylabel": f"Mean Experience Value"},
    "cost": {
        "placement": 223,
        "graphs": [{"color": "red_transp"}, {"color": "red"}],
        "ylabel": f"Average Brain Cost",
        "xlabel": f"Games Played",
    },
    "weights": {
        "placement": 224,
        "graphs": [{"color": "blue"}],
        "ylabel": f"Weights Range",
        "xlabel": f"Games Played",
    },
}
plotter = Plotter("Policy Network Performance", plot_data)

# Before learning, just get random data in the experience buffer
print(f"Playing rounds of {RANDOM_GAME_COUNT} games for experience")
while learner_robot.experience_buffer_usage < 0.2:
    random_game.play(RANDOM_GAME_COUNT)
    print(f"Experience Buffer Usage: {learner_robot.experience_buffer_usage * 100:5.1f}%")
print()

game_count = 0
prev_score = None
prev_mean_experience_value = None
brain_cost = 0
brain_cost_ema = None
best_score = -1e6
running = True
while running:
    try:

        # Train and learn
        print("Training...")
        trainer_robot.epsilon = np.random.rand() * 0.2
        train_game.play(TRAIN_GAME_COUNT)
        print("Learning...")

        brain_cost = 0
        for i in range(BATCH_ITERATIONS):
            learner_robot.learn(1)
            brain_cost += robot_brain.cost
        brain_cost /= BATCH_ITERATIONS
        brain_cost_ema = (
            0.9 * brain_cost_ema + 0.1 * brain_cost if brain_cost_ema and not np.isnan(brain_cost_ema) else brain_cost
        )

        # Try against a random player
        print("Playing...")
        random_game.reset_score()
        random_game.play(RANDOM_GAME_COUNT)
        game_count += RANDOM_GAME_COUNT

        score_tuple = random_game.score
        score_tuple_rel = score_tuple[0] / RANDOM_GAME_COUNT * 100, score_tuple[1] / RANDOM_GAME_COUNT * 100
        score = score_tuple_rel[0] - score_tuple_rel[1]
        score_diff = (score - prev_score) / abs(prev_score) * 100 if prev_score else 0
        prev_score = score

        mean_experience_value = learner_robot.mean_experience_value
        mean_experience_value_diff = (
            (mean_experience_value - prev_mean_experience_value) / abs(prev_mean_experience_value) * 100
            if prev_mean_experience_value
            else 0.0
        )
        prev_mean_experience_value = mean_experience_value

        weight_range = robot_brain.weight_range

        # Useful info
        print(
            f"Games Played: {game_count}\n"
            f"Score: {score:5.1f}% {score_diff:+4.1f}% {'HIGH SCORE!' if score > best_score else ''}\n"
            f"Wins / Losses: {score_tuple_rel[0]:.1f}% / {score_tuple_rel[1]:.1f}%\n"
            f"Mean Experience Value: {mean_experience_value:6.3f} {mean_experience_value_diff:+4.1f}%\n"
            f"Experience Buffer Usage: {learner_robot.experience_buffer_usage * 100:5.1f}%\n"
            f"Brain Cost: {brain_cost:4.3f}\n"
            f"Brain Cost EMA: {(0 if brain_cost_ema is None else brain_cost_ema):4.3f}\n"
            f"Weight Range: [{weight_range[0]:6.3f}, {weight_range[1]:6.3f}]\n"
            f"Output: {robot_brain.output[0,:]}\n"
            f"Target: {robot_brain.target[0,:]}\n"
        )

        # Update plot data
        game_counts.append(game_count)
        wins.append(score_tuple_rel[0])
        losses.append(score_tuple_rel[1])
        scores.append(score)
        mean_experience_values.append(mean_experience_value)
        brain_costs.append(brain_cost)
        brain_costs_ema.append(brain_cost_ema)
        weight_ranges.append(weight_range[1] - weight_range[0])
        graph_data = {
            "score": (scores, losses, wins, game_counts),
            "value": (mean_experience_values, game_counts),
            "cost": (brain_costs, brain_costs_ema, game_counts),
            "weights": (weight_ranges, game_counts),
        }
        plotter.update_data(graph_data)

        # Save the brain on all time high scores
        if score > best_score:
            best_score = score
            pickle.dump(robot_brain, open(BRAIN_FILEPATH, "wb"))
            plotter.save_image(f"plots/connect-four-training.png")

    except KeyboardInterrupt:
        running = False


plotter.save_image(f"plots/connect-four-training.png")
