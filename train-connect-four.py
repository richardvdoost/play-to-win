import pickle

import numpy as np

from brain import Brain
from brain.activation_functions import ReLU, Softmax
from games import ConnectFour
from players import PolicyGradientPlayer
from plotter import Plotter

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

TRAIN_GAME_COUNT = 50
MAX_ROUNDS_WITHOUT_HIGHSCORE = 32

DISCOUNT_RATE = 0.8
EXPERIENCE_BATCH_SIZE = 512
BATCH_ITERATIONS = 1
EXPERIENCE_BUFFER_SIZE = 2 ** 17

LEARNING_RATE = 0.0001
REGULARIZATION = 0.1

BRAIN_TOPOLOGY = (
    (84, None),
    (2048, ReLU),
    (2048, ReLU),
    (2048, ReLU),
    (7, Softmax),
)
# 8,581,127 parameters
TRAINER_BRAIN_FILEPATH = "brain/saved/connect-four-trainer-brain.pickle"
LEARNER_BRAIN_FILEPATH = "brain/saved/connect-four-learner-brain.pickle"

print("\nSTART CONNECTFOUR GAME TRAINING SESSION\n")

try:
    learner_brain = pickle.load(open(LEARNER_BRAIN_FILEPATH, "rb"))
    print("Loaded previously best brain for the learner robot")
except Exception:
    learner_brain = Brain(BRAIN_TOPOLOGY, learning_rate=LEARNING_RATE, regularization=REGULARIZATION)
    print("Created brain from scratch for the learner robot")

try:
    trainer_a_brain = pickle.load(open(TRAINER_BRAIN_FILEPATH, "rb"))
    print("Loaded previously best brain for the trainer robot")
except Exception:
    trainer_a_brain = Brain(BRAIN_TOPOLOGY, learning_rate=LEARNING_RATE, regularization=REGULARIZATION)
    print("Created brain from scratch for the trainer robot")

trainer_b_brain = Brain(BRAIN_TOPOLOGY, learning_rate=LEARNING_RATE, regularization=REGULARIZATION)

# The robots
learner_robot = PolicyGradientPlayer(
    learner_brain,
    discount_rate=DISCOUNT_RATE,
    batch_iterations=BATCH_ITERATIONS,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)
trainer_robot_a = PolicyGradientPlayer(
    trainer_a_brain,
    discount_rate=DISCOUNT_RATE,
    batch_iterations=BATCH_ITERATIONS,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)
trainer_robot_b = PolicyGradientPlayer(
    trainer_b_brain,
    discount_rate=DISCOUNT_RATE,
    batch_iterations=BATCH_ITERATIONS,
    experience_batch_size=EXPERIENCE_BATCH_SIZE,
    experience_buffer_size=EXPERIENCE_BUFFER_SIZE,
)
learner_robot.learn_while_playing = True
trainer_robot_a.learn_while_playing = True
trainer_robot_b.learn_while_playing = True

game_a = ConnectFour((learner_robot, trainer_robot_a))
game_b = ConnectFour((learner_robot, trainer_robot_b))

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

game_count = 0
prev_score = prev_trainer_a_score = prev_trainer_b_score = None
prev_mean_experience_value = None
win_ema = None
lose_ema = None
score_ema = None
brain_cost = 0
brain_cost_ema = None
best_trainer_a_score = best_trainer_b_score = -TRAIN_GAME_COUNT
rounds_without_highscore = 0
running = True
while running:
    try:

        game_a.reset_score()
        game_a.play(TRAIN_GAME_COUNT)
        game_b.reset_score()
        game_b.play(TRAIN_GAME_COUNT)

        brain_cost = learner_brain.cost
        brain_cost_ema = (
            0.9 * brain_cost_ema + 0.1 * brain_cost if brain_cost_ema and not np.isnan(brain_cost_ema) else brain_cost
        )
        brain_deltas = "\n".join([f"{learner_brain.output[i, :] - learner_brain.target[i, :]}" for i in range(3)])

        game_count += TRAIN_GAME_COUNT * 2
        score_tuple = [game_a.score[i] + game_b.score[i] for i in range(2)]
        score_tuple_rel = score_tuple[0] / TRAIN_GAME_COUNT * 50, score_tuple[1] / TRAIN_GAME_COUNT * 50
        score = score_tuple_rel[0] - score_tuple_rel[1]
        score_diff = (score - prev_score) / abs(prev_score) * 100 if prev_score else 0
        prev_score = score
        trainer_a_score = game_a.score[1] - game_a.score[0]
        trainer_b_score = game_b.score[1] - game_b.score[0]
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
            f"Score:        {score:5.1f}% {score_diff:+4.1f}%\n"
            f"Game A Score: {game_a.score}\n"
            f"Game B Score: {game_b.score}\n"
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

        # Keep track of high scores
        highscore = False
        if trainer_a_score > best_trainer_a_score:
            print("Trainer A has a high score")
            best_trainer_a_score = trainer_a_score
            highscore = True

        if trainer_b_score > best_trainer_b_score:
            print("Trainer B has a high score")
            best_trainer_b_score = trainer_b_score
            highscore = True

        if highscore:
            rounds_without_highscore = 0
            plotter.save_image("plots/connect-four-training.png")

        else:
            rounds_without_highscore += 1
            print(f"Rounds without highscore: {rounds_without_highscore}")

            # If we hit the max rounds of no highscore, and we already had a positive score, stop
            if rounds_without_highscore > MAX_ROUNDS_WITHOUT_HIGHSCORE:
                print(" - Too long without highscore, exiting...")
                break

    except KeyboardInterrupt:
        running = False

print("\nTRAINING SESSION DONE\n")

# Save the learner brain and the best trainer brain
pickle.dump(learner_brain, open(LEARNER_BRAIN_FILEPATH, "wb"))
pickle.dump(
    trainer_a_brain if trainer_a_score > trainer_b_score else trainer_b_brain, open(TRAINER_BRAIN_FILEPATH, "wb")
)

plotter.save_image("plots/connect-four-training.png")
