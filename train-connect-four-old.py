import pickle

import numpy as np

from brain import Brain
from brain.activation_functions import ReLU, Sigmoid, Softmax
from games import ConnectFour
from players import PolicyGradientPlayer
from plotter import Plotter

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

TRAIN_GAME_COUNT = 500
MAX_ROUNDS_WITHOUT_HIGHSCORE = 20
MIN_BUFFER_USAGE_BEFORE_LEARNING = 0.05

STARTING_EPSILON = 0.4
FINAL_EPSILON = 0.01
DISCOUNT_RATE = 0.8
EXPERIENCE_BATCH_SIZE = 512
BATCH_ITERATIONS = 1
EXPERIENCE_BUFFER_SIZE = 2 ** 17

LEARNING_RATE = 0.0001
REGULARIZATION = 0.1

BRAIN_TOPOLOGY = (
    (84, None),
    (256, ReLU),
    (7, Softmax),
)
TRAINER_BRAIN_FILEPATH = "brain/saved/connect-four-brain.pickle"

print("\nSTART CONNECTFOUR GAME TRAINING SESSION\n")

adaptive_trainer_brain = Brain(BRAIN_TOPOLOGY, learning_rate=LEARNING_RATE, regularization=REGULARIZATION)
print("Created brain from scratch for the adaptive trainer robot")
try:
    static_trainer_brain = pickle.load(open(TRAINER_BRAIN_FILEPATH, "rb"))
    print("Loaded previously best brain for the static trainer robot")
except Exception:
    static_trainer_brain = Brain(BRAIN_TOPOLOGY, learning_rate=LEARNING_RATE, regularization=REGULARIZATION)
    print("Created brain from scratch for the static trainer robot")

static_trainer_brain_hidden_layers = static_trainer_brain.neuron_layers[1:-1]
hidden_layer_count = len(static_trainer_brain_hidden_layers)
neuron_count = 0
for layer in static_trainer_brain_hidden_layers:
    neuron_count += len(layer)
neurons_per_layer = neuron_count / hidden_layer_count
neurons_per_layer *= 1.05
if neurons_per_layer > 2048:
    hidden_layer_count += 1
    neurons_per_layer = neuron_count / hidden_layer_count

learner_brain_topology = (
    [BRAIN_TOPOLOGY[0]] + [(int(neurons_per_layer), ReLU)] * hidden_layer_count + [BRAIN_TOPOLOGY[-1]]
)
learner_brain = Brain(learner_brain_topology, learning_rate=LEARNING_RATE, regularization=REGULARIZATION)
print(f"Created brain from scratch for the player robot - topology: {learner_brain_topology}")

# The robots
learner_robot = PolicyGradientPlayer(
    learner_brain,
    discount_rate=DISCOUNT_RATE,
    batch_iterations=BATCH_ITERATIONS,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)
adaptive_trainer_robot = PolicyGradientPlayer(
    learner_brain,
    discount_rate=DISCOUNT_RATE,
    batch_iterations=BATCH_ITERATIONS,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)
static_trainer_robot = PolicyGradientPlayer(
    static_trainer_brain,
    discount_rate=DISCOUNT_RATE,
    batch_iterations=BATCH_ITERATIONS,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)

adaptive_train_game = ConnectFour((learner_robot, adaptive_trainer_robot))
static_train_game = ConnectFour((learner_robot, static_trainer_robot))

# Initialize plot data
game_counts = []
wins = []
wins_ema = []
losses = []
losses_ema = []
scores = []
scores_ema = []
mean_experience_values = []
mean_confidences = []
brain_costs = []
brain_costs_ema = []
weight_ranges = []
weight_means = []
plot_data = {
    "score": {
        "placement": 221,
        "graphs": [
            {"color": "blue_transp"},
            {"label": "Score", "color": "blue"},
            {"color": "red_transp"},
            {"label": "% Losses", "color": "red"},
            {"color": "green_transp"},
            {"label": "% Wins", "color": "green"},
        ],
        "ylabel": f"Average of {TRAIN_GAME_COUNT} Games",
        "legend": True,
    },
    "experience": {
        "placement": 222,
        "graphs": [{"color": "green", "label": "State Value"}, {"color": "blue", "label": "Action Confidence"}],
        "ylabel": "Mean Experience Value",
        "legend": True,
    },
    "cost": {
        "placement": 223,
        "graphs": [{"color": "red_transp"}, {"color": "red"}],
        "ylabel": "Average Brain Cost",
        "xlabel": "Games Played",
    },
    "weights": {
        "placement": 224,
        "graphs": [{"color": "blue", "label": "Abs. Max"}, {"color": "green", "label": "Abs. Mean"},],
        "ylabel": "Absolute Weight Value",
        "xlabel": "Games Played",
        "legend": True,
    },
}
plotter = Plotter("Policy Network Performance", plot_data)

# Before learning, just get random data in the experience buffer
print(
    f"\nPlaying warmup rounds of {TRAIN_GAME_COUNT} games to fill the "
    f"experience buffer to at least {MIN_BUFFER_USAGE_BEFORE_LEARNING * 100:4.1f}%"
)
learner_robot.epsilon = STARTING_EPSILON
static_trainer_robot.epsilon = STARTING_EPSILON
while learner_robot.experience_buffer_usage < MIN_BUFFER_USAGE_BEFORE_LEARNING:
    static_train_game.play(TRAIN_GAME_COUNT)
    print(f"Experience Buffer Usage: {learner_robot.experience_buffer_usage * 100:4.1f}%")

# Give the adaptive trainer robot a lot of experiences to learn from
adaptive_trainer_robot.experiences = learner_robot.experiences + static_trainer_robot.experiences

learner_robot.learn_while_playing = True
adaptive_trainer_robot.learn_while_playing = True

game_count = 0
prev_score = None
prev_mean_experience_value = None
win_ema = None
lose_ema = None
score_ema = None
brain_cost = 0
brain_cost_ema = None
best_score = -100
best_static_train_score = -100
rounds_without_highscore = 0
running = True
while running:
    try:

        epsilon = 1 / (1 + game_count / TRAIN_GAME_COUNT * 0.05) * (STARTING_EPSILON - FINAL_EPSILON) + FINAL_EPSILON

        # Learn and play
        print(f"Playing against both trainer bots using epsilon: {epsilon:.4f}")
        learner_robot.epsilon = epsilon
        adaptive_trainer_robot.epsilon = epsilon
        static_trainer_robot.epsilon = epsilon

        static_train_game.reset_score()
        static_train_game.play(int(TRAIN_GAME_COUNT / 2))

        learner_robot.epsilon /= 2
        adaptive_train_game.reset_score()
        adaptive_train_game.play(int(TRAIN_GAME_COUNT / 2))

        brain_cost = learner_brain.cost
        brain_cost_ema = (
            0.9 * brain_cost_ema + 0.1 * brain_cost if brain_cost_ema and not np.isnan(brain_cost_ema) else brain_cost
        )
        brain_deltas = "\n".join([f"{learner_brain.output[i, :] - learner_brain.target[i, :]}" for i in range(3)])

        game_count += TRAIN_GAME_COUNT
        score_tuple = [static_train_game.score[i] + adaptive_train_game.score[i] for i in range(2)]
        score_tuple_rel = score_tuple[0] / TRAIN_GAME_COUNT * 100, score_tuple[1] / TRAIN_GAME_COUNT * 100
        score = score_tuple_rel[0] - score_tuple_rel[1]
        score_diff = (score - prev_score) / abs(prev_score) * 100 if prev_score else 0
        prev_score = score
        static_train_score = (static_train_game.score[0] - static_train_game.score[1]) / TRAIN_GAME_COUNT * 50
        win_ema = 0.9 * win_ema + 0.1 * score_tuple_rel[0] if win_ema else score_tuple_rel[0]
        lose_ema = 0.9 * lose_ema + 0.1 * score_tuple_rel[1] if lose_ema else score_tuple_rel[1]
        score_ema = 0.9 * score_ema + 0.1 * score if score_ema else score

        mean_experience_value = learner_robot.mean_experience_value
        mean_experience_value_diff = (
            (mean_experience_value - prev_mean_experience_value) / abs(prev_mean_experience_value) * 100
            if prev_mean_experience_value
            else 0.0
        )
        prev_mean_experience_value = mean_experience_value
        mean_confidence = learner_robot.confidence

        synapse_stats = learner_brain.synapse_stats

        # Useful info
        print(
            f"Games Played: {game_count}\n"
            f"Score:                   {score:5.1f}% {score_diff:+4.1f}% {'HIGHSCORE!' if score > best_score else ''}\n"
            f"Static Training Score:   {static_train_game.score} {'HIGHSCORE!' if static_train_score > best_static_train_score > 0 else ''}\n"
            f"Adaptive Training Score: {adaptive_train_game.score}\n"
            f"Wins: {score_tuple_rel[0]:5.1f}% - Losses: {score_tuple_rel[1]:5.1f}%\n"
            f"Mean Experience Value: {mean_experience_value:6.3f} {mean_experience_value_diff:+4.1f}%\n"
            f"Experience Buffer Usage: {learner_robot.experience_buffer_usage * 100:5.1f}%\n"
            f"Brain Cost: {brain_cost:4.3f}\n"
            f"Brain Cost EMA: {(0 if brain_cost_ema is None else brain_cost_ema):4.3f}\n"
            f"Synapse Stats: {synapse_stats}\n"
            f"Last 3 Brain Output Deltas:\n{brain_deltas}\n"
        )

        # Update plot data
        game_counts.append(game_count)
        wins.append(score_tuple_rel[0])
        wins_ema.append(win_ema)
        losses.append(score_tuple_rel[1])
        losses_ema.append(lose_ema)
        scores.append(score)
        scores_ema.append(score_ema)
        mean_experience_values.append(mean_experience_value)
        mean_confidences.append(mean_confidence)
        brain_costs.append(brain_cost)
        brain_costs_ema.append(brain_cost_ema)
        weight_ranges.append(synapse_stats["weight_range"])
        weight_means.append(synapse_stats["weight_mean"])
        graph_data = {
            "score": (scores, scores_ema, losses, losses_ema, wins, wins_ema, game_counts),
            "experience": (mean_experience_values, mean_confidences, game_counts),
            "cost": (brain_costs, brain_costs_ema, game_counts),
            "weights": (weight_ranges, weight_means, game_counts),
        }
        plotter.update_data(graph_data)

        # Save the brain on all time high scores
        if score > best_score:
            best_score = score
            rounds_without_highscore = 0
            plotter.save_image("plots/connect-four-training.png")

            # If the score is better than the previously saved brain, save it
            if score > 0 and static_train_score > best_static_train_score:
                best_static_train_score = static_train_score
                pickle.dump(learner_brain, open(TRAINER_BRAIN_FILEPATH, "wb"))
        else:
            rounds_without_highscore += 1

            print(f"Rounds without highscore: {rounds_without_highscore}")

            # If we hit the max rounds of no highscore, and we already had a positive score, stop
            if rounds_without_highscore > MAX_ROUNDS_WITHOUT_HIGHSCORE and best_static_train_score > 0:
                print(" - Too long without highscore, exiting...")
                break

    except KeyboardInterrupt:
        running = False

print("\nTRAINING SESSION DONE\n")
plotter.save_image("plots/connect-four-training.png")
