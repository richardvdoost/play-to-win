import numpy as np
import pickle

from brain import Brain
from brain.activation_functions import Identity, ReLU, Sigmoid, Softplus

from games import TicTacToe
from players import RandomPlayer, PolicyGradientPlayer

# Set some NumPy print options
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# brain_topology = (
#     (2, None),
#     (2, Softplus),
#     (1, Sigmoid),
# )

# training_data = (
#     {"input": (0, 0), "target": (0,)},
#     {"input": (1, 0), "target": (1,)},
#     {"input": (0, 1), "target": (1,)},
#     {"input": (1, 1), "target": (0,)},
# )

# my_brain = Brain(brain_topology, learning_rate=0.4, momentum=0.8)
# my_brain.train(training_data, 1000)

# print(my_brain)


##################

brain_topology = (
    (18, None),
    (48, Softplus),
    (42, Softplus),
    (9, Sigmoid),
)

try:
    # player_brain = pickle.load(open("brain/saved/player_brain_new.pickle", "rb"))
    pre_trained_brain = pickle.load(open("brain/saved/tictactoe_good.pickle", "rb"))
    opponent = PolicyGradientPlayer(pre_trained_brain)
    opponent.is_learning = False
except Exception:
    opponent = RandomPlayer()
player_brain = Brain(brain_topology, learning_rate=0.01, momentum=0.9, regularization=None)

policy_player = PolicyGradientPlayer(
    player_brain,
    discount_factor=0.90,
    reward_scale=8,
    mini_batch_size=256,
    train_iterations=1,
    experience_buffer_size=50000,
)

tictactoe_game = TicTacToe((policy_player, opponent))

update_every = 200
game_count = 0
prev_score = None
prev_mean_experience_value = None
perfect_score_count = 0
running = True
while running:
    try:
        tictactoe_game.play(update_every)
        game_count += update_every

        score_tuple = tictactoe_game.score
        score = score_tuple[0] - score_tuple[1]
        score_diff = (score - prev_score) / abs(prev_score) * 100 if prev_score else 0
        prev_score = score

        mean_experience_value = policy_player.mean_experience_value
        mean_experience_value_diff = (
            (mean_experience_value - prev_mean_experience_value) / abs(prev_mean_experience_value) * 100
            if prev_mean_experience_value
            else 0.0
        )
        prev_mean_experience_value = mean_experience_value

        print(
            f"Games Played: {game_count}\n"
            f"Score: {score_tuple} -> {score:3d} {score_diff:+4.1f}%\n"
            f"Mean Experience Value: {mean_experience_value:6.3f} {mean_experience_value_diff:+4.1f}%\n"
            f"Experience Buffer Usage: {policy_player.experience_buffer_usage * 100:5.1f}%\n"
            f"Brain Cost: {player_brain.cost:4.3f}\n"
            f"Weight Range: [{player_brain.weight_range[0]:6.3f}, {player_brain.weight_range[1]:6.3f}]\n"
            f"Output: {player_brain.output[0,:]}\n"
            f"Target: {player_brain.target[0,:]}\n"
        )
        tictactoe_game.reset_score()

        # Stop when we have a perfect score (0 losses)
        if score_tuple[1] == 0:
            perfect_score_count += 1
            if perfect_score_count > 3:
                print("Played perfectly for 3 streaks! You can stop now :)")
                running = False

    except KeyboardInterrupt:
        running = False

pickle.dump(player_brain, open("brain/saved/player_brain_new.pickle", "wb"))
