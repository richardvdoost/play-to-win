import pickle
import pprint
import time

import numpy as np

from brain import Brain
from brain.activation_functions import ReLU, Sigmoid, Softmax, Softplus
from games import TicTacToe
from players import PolicyGradientPlayer, RandomPlayer
from plotter import Plotter

GENERATION_SIZE = 10
TRAIN_TIME = 10
PLAY_COUNT = 1000

activation_functions = (ReLU, Sigmoid, Softplus)

random_player = RandomPlayer()
# trained_brain = pickle.load(open("brain/saved/player_brain-512-512_200-in-a-row.pickle", "rb",))
# trained_player = PolicyGradientPlayer(trained_brain)
# trained_player.is_learning = False

# Create base / origin model
generation = [
    {
        "discount_factor": 0.5,
        "reward_factor": 1,
        "experience_batch_size": 1,
        "experience_buffer_size": 1,
        "batch_iterations": 1,
        "learning_rate": 1,
        "regularization": 1,
        "brain": [],
        "fitness": 0,
    },
]

pp = pprint.PrettyPrinter(indent=4)

# Create a plot figure
plot_data = {
    "score_random": {
        "placement": 211,
        "graphs": [
            {"label": "Min/Avg/Max Losses", "color": "red"},
            {"color": "red"},
            {"color": "red"},
            # {"label": "Min/Avg/Max Score", "color": "blue"},
            # {"label": "Min/Avg/Max Score", "color": "blue"},
            # {"label": "Min Losses", "color": "red"},
            # {"label": "Max Wins", "color": "green"},
        ],
        "ylabel": f"Score against random player {PLAY_COUNT} games",
        "legend": True,
    },
    "brain_size": {
        "placement": 212,
        "graphs": [{"color": "blue"}, {"color": "blue"}, {"color": "blue"}],
        "ylabel": f"Brain Size (parameters)",
        "xlabel": f"Generation",
    },
}
plotter = Plotter("Generation Performance", plot_data)


def brain_size(hidden_layers):

    layer_sizes = [18] + hidden_layers + [9]
    parameter_count = 0
    for size_a, size_b in zip(layer_sizes[:-1], layer_sizes[1:]):
        parameter_count += (size_a + 1) * size_b
    return parameter_count


running = True
generation_index = 0
# scores_min = []
# scores_avg = []
# scores_max = []
losses_min = []
losses_avg = []
losses_max = []
# wins_max = []
brain_size_min = []
brain_size_avg = []
brain_size_max = []
while running:
    try:

        # Take the generation fitness and create a softmax distribution to sample over
        generation_fitness = np.array([candidate["fitness"] for candidate in generation])
        t = np.exp(generation_fitness * 0.5)
        fitness_softmax = t / np.sum(t)
        print(f"Fitness softmax: {fitness_softmax}")

        # Create a new generation by sampling individual parents from the old generation
        generation_index += 1
        print(f"Creating generation {generation_index}")
        new_generation = []
        for _ in range(GENERATION_SIZE):

            # Pick/sample a parent based on their fitness
            choice = np.random.choice(len(fitness_softmax), p=fitness_softmax)
            parent = generation[choice]
            print(f" - Picked parent {choice} with fitness: {parent['fitness']}")

            # Create a child based on the parent
            child = {
                "discount_factor": Sigmoid.activate(
                    np.log(parent["discount_factor"] / (1 - parent["discount_factor"])) + np.random.normal(scale=0.05)
                ),
                "reward_factor": parent["reward_factor"] * np.exp(np.random.normal(scale=0.05)),
                "experience_batch_size": int(
                    2 ** round(np.log2(parent["experience_batch_size"]) + max(0, np.random.normal(scale=0.4)))
                ),
                "experience_buffer_size": int(
                    2 ** round(np.log2(parent["experience_buffer_size"]) + max(0, np.random.normal(scale=0.5)))
                ),
                "batch_iterations": int(max(1, round(parent["batch_iterations"] + np.random.normal(scale=0.4)))),
                "learning_rate": parent["learning_rate"] * np.exp(np.random.normal(scale=0.05)),
                "regularization": parent["regularization"] * np.exp(np.random.normal(scale=0.05)),
            }

            # Evolve the brain
            brain = parent["brain"][:]
            brain_growth = min(max(-1, round(np.random.normal(scale=0.3))), 1)
            if brain_growth == 1:
                activation_function = activation_functions[np.random.choice(len(activation_functions))]

                if len(parent["brain"]) == 0:
                    print("   Growing a brain!")
                    # Spawn a new brian layer out of nothing, use the base layer
                    brain = [[14, activation_function]]  # Hardcoded 9 because tictactoe

                else:
                    # Pick a random brain layer and double it, but choose a random activation function
                    print("   Getting a bigger brain!")
                    brain = parent["brain"][:]
                    layer_index = np.random.choice(len(brain))
                    new_layer = [brain[layer_index][0], activation_function]
                    brain.insert(layer_index, new_layer)

            elif brain_growth == -1:
                if len(parent["brain"]) > 0:
                    print("   Losing brain cells!")
                    # Remove a random brain layer
                    del brain[np.random.choice(len(brain))]

            # Evolve the brain layer sizes
            for i, layer in enumerate(brain):
                brain[i] = [int(round(layer[0] * np.exp(np.random.normal(scale=0.1)))), layer[1]]

            child["brain"] = brain

            new_generation.append(child)

        generation = new_generation

        print("Training and playing")
        players = []
        play_scores = []
        for i, candidate in enumerate(generation):
            print(f"Candidate {i} {[layer[0] for layer in candidate['brain']]}")

            # Create a brain
            brain_topology = [(18, None)] + candidate["brain"] + [(9, Softmax)]
            brain = Brain(
                brain_topology, learning_rate=candidate["learning_rate"], regularization=candidate["regularization"]
            )

            # Create a player
            policy_player = PolicyGradientPlayer(
                brain,
                discount_factor=candidate["discount_factor"],
                reward_factor=candidate["reward_factor"],
                batch_iterations=candidate["batch_iterations"],
                experience_batch_size=candidate["experience_batch_size"],
                experience_buffer_size=candidate["experience_buffer_size"],
            )
            players.append(policy_player)

            # Create the games
            random_game = TicTacToe((policy_player, random_player))
            # trained_game = TicTacToe((policy_player, trained_player))

            # Start the timer and train for the given amount of time
            print(f" - Start training for {TRAIN_TIME} seconds")
            start_time = time.time()
            while time.time() < start_time + TRAIN_TIME:
                random_game.play(100)
                # trained_game.play(100)

            # Play / compete against a random player, and a pre trained player
            print(f" - Start playing {PLAY_COUNT} games")
            policy_player.is_learning = False
            policy_player.act_greedy = True
            random_game.reset_score()
            random_game.play(int(PLAY_COUNT))
            # trained_game.reset_score()
            # trained_game.play(int(PLAY_COUNT))
            policy_player.act_greedy = False

            # Save the fitness
            score = (
                random_game.score
            )  # random_game.score[0] + trained_game.score[0], random_game.score[1] + trained_game.score[1]
            fitness = -score[1]
            print(f" - Fitness: {fitness} (wins: {score[0]}, losses: {score[1]})")
            candidate["fitness"] = fitness
            play_scores.append(score)

        # Play against each other
        # print("Competition")
        # for i in range(len(players) - 1):
        #     for j in range(i + 1, len(players)):
        #         tictactoe_game = TicTacToe((players[i], players[j]))
        #         tictactoe_game.play(int(PLAY_COUNT / GENERATION_SIZE - 1))
        #         score = tictactoe_game.score

        #         print(f"{i} against {j} - score {score}")
        #         generation[i]["fitness"] += score[0]
        #         generation[j]["fitness"] += score[1]

        # for i, candidate in enumerate(generation):
        #     print(f"Candidate {i} {[layer[0] for layer in candidate['brain']]} - Fitness: {candidate['fitness']}")

        print(f"\nGENERATION {generation_index} RESULTS:")
        pp.pprint(generation)
        print("\n")

        # scores = np.array([score[0] - score[1] for score in play_scores])
        # wins = np.array([score[0] for score in play_scores])
        losses = np.array([score[1] for score in play_scores])
        brains = [[layer[0] for layer in candidate["brain"]] for candidate in generation]
        brain_sizes = np.array([brain_size(layers) for layers in brains])

        # scores_min.append(scores.min())
        # scores_avg.append(scores.mean())
        # scores_max.append(scores.max())
        losses_min.append(losses.min())
        losses_avg.append(losses.mean())
        losses_max.append(losses.max())
        # wins_max.append(wins.max())

        brain_size_min.append(brain_sizes.min())
        brain_size_avg.append(brain_sizes.mean())
        brain_size_max.append(brain_sizes.max())

        generations = list(range(1, generation_index + 1))

        graph_data = {
            "score_random": (losses_min, losses_avg, losses_max, generations),
            "brain_size": (brain_size_min, brain_size_avg, brain_size_max, generations),
        }
        plotter.update_data(graph_data)

    except KeyboardInterrupt:
        running = False

plotter.save_image(f"plots/evolution.png")
