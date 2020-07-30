import pickle
import pprint
import time
from multiprocessing import Pool

import numpy as np

from brain import Brain
from brain.activation_functions import ReLU, Sigmoid, Softmax, Softplus
from games import TicTacToe
from players import PolicyGradientPlayer, RandomPlayer
from plotter import Plotter

GENERATION_SIZE = 8
TRAIN_TIME = GENERATION_SIZE * 120
PLAY_COUNT = 1000

activation_functions = (ReLU, Sigmoid, Softplus)

random_player = RandomPlayer()


# Create base / origin model
generation = [
    {
        "discount_factor": 0.5,
        "reward_factor": 1.5,
        "experience_batch_size": 512,
        "experience_buffer_size": 2 ** 16,
        "batch_iterations": 1,
        "learning_rate": 0.001,
        "regularization": 0.15,
        "brain": [[256, ReLU]],
        "fitness": 0,
    },
]

# Create a plot figure
plot_data = {
    "score_random": {
        "placement": 121,
        "graphs": [
            {"label": "Min/Avg/Max Losses", "color": "red"},
            {"color": "red"},
            {"color": "red"},
        ],
        "ylabel": f"Out of {PLAY_COUNT} games",
        "legend": True,
        "xlabel": f"Generation",
    },
    "brain_size": {
        "placement": 122,
        "graphs": [{"color": "blue"}, {"color": "blue"}, {"color": "blue"}],
        "ylabel": f"Brain Size (# of synapses)",
        "xlabel": f"Generation",
    },
}
plotter = Plotter(
    f"Generation Performance - {TRAIN_TIME / GENERATION_SIZE} seconds of training per player",
    plot_data,
)


def train(game):
    global TRAIN_TIME
    games_per_step = 50
    games_played = 0
    start_time = time.time()
    while time.time() < start_time + TRAIN_TIME:
        game.play(games_per_step)
        games_played += games_per_step

    print(f" - Trained on {games_played} games")
    return game


def play(game):
    global PLAY_COUNT
    game.reset_score()
    game.players[0].is_learning = False
    game.players[0].act_greedy = True
    game.play(PLAY_COUNT)
    return game


def brain_size(hidden_layers):

    layer_sizes = [18] + hidden_layers + [9]
    parameter_count = 0
    for size_a, size_b in zip(layer_sizes[:-1], layer_sizes[1:]):
        parameter_count += (size_a + 1) * size_b
    return parameter_count


def wait(processes):
    for process in processes:
        process.join()


def print_genome(genome):
    print("{")
    for gene, value in genome.items():
        value = f"{value:.5f}" if isinstance(value, float) else value
        print(f'    "{gene}": {value},')
    print("}\n")


np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

generation_index = 0
losses_min = []
losses_avg = []
losses_max = []
brain_size_min = []
brain_size_avg = []
brain_size_max = []
best_genome = {"fitness": -1e6}

if __name__ == "__main__":
    running = True
    while running:
        try:

            # Take the generation fitness and create a softmax distribution to sample over
            generation_fitness = np.array(
                [candidate["fitness"] for candidate in generation]
            )
            t = np.exp(generation_fitness)
            fitness_softmax = t / np.sum(t)
            print(f"Fitness softmax: {fitness_softmax}\n")

            # Create a new generation by sampling individual parents from the old generation
            generation_index += 1
            print(f"Creating generation {generation_index}")
            new_generation = []
            for _ in range(GENERATION_SIZE):

                # Pick/sample a parent based on their fitness
                choice = np.random.choice(len(fitness_softmax), p=fitness_softmax)
                parent = generation[choice]
                print(
                    f" - Picked parent {choice:02d} with fitness: {parent['fitness']:.1f}"
                )

                # Create a child based on the parent
                child = {
                    "discount_factor": Sigmoid.activate(
                        np.log(
                            parent["discount_factor"] / (1 - parent["discount_factor"])
                        )
                        + np.random.normal(scale=0.05)
                    ),
                    "reward_factor": parent["reward_factor"]
                    * np.exp(np.random.normal(scale=0.05)),
                    "experience_batch_size": int(
                        min(
                            2 ** 20,
                            2
                            ** round(
                                np.log2(parent["experience_batch_size"])
                                + max(0, np.random.normal(scale=0.4))
                            ),
                        )
                    ),
                    "experience_buffer_size": int(
                        min(
                            2 ** 20,
                            2
                            ** round(
                                np.log2(parent["experience_buffer_size"])
                                + max(0, np.random.normal(scale=0.5))
                            ),
                        )
                    ),
                    "batch_iterations": int(
                        max(
                            1,
                            round(
                                parent["batch_iterations"] + np.random.normal(scale=0.4)
                            ),
                        )
                    ),
                    "learning_rate": parent["learning_rate"]
                    * np.exp(np.random.normal(scale=0.05)),
                    "regularization": parent["regularization"]
                    * np.exp(np.random.normal(scale=0.05)),
                }

                # Evolve the brain
                brain = parent["brain"][:]
                brain_growth = min(max(-1, round(np.random.normal(scale=0.3))), 1)
                if brain_growth == 1:
                    activation_function = activation_functions[
                        np.random.choice(len(activation_functions))
                    ]

                    if len(parent["brain"]) == 0:
                        print("   Growing a brain!")
                        # Spawn a brain layer out of nothing, use the base layer
                        brain = [
                            [14, activation_function]
                        ]  # Hardcoded 14 because tictactoe average of input neurons and output neurons

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
                    brain[i] = [
                        int(round(layer[0] * np.exp(np.random.normal(scale=0.1)))),
                        layer[1],
                    ]

                child["brain"] = brain

                new_generation.append(child)

            generation = new_generation
            print()

            play_scores = []
            games = []
            for i, candidate in enumerate(generation):

                # Create a brain
                brain_topology = [(18, None)] + candidate["brain"] + [(9, Softmax)]
                brain = Brain(
                    brain_topology,
                    learning_rate=candidate["learning_rate"],
                    regularization=candidate["regularization"],
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

                # Create the game
                random_game = TicTacToe((policy_player, random_player))
                random_game.id = i
                games.append(random_game)

            print(f"Training all players for {TRAIN_TIME} seconds")
            with Pool(GENERATION_SIZE) as pool:
                games = pool.map(train, games)
            print("Done training\n")

            print(f"Let all players play {PLAY_COUNT} games each")
            with Pool(GENERATION_SIZE) as pool:
                games = pool.map(play, games)
            print("Done playing\n")

            for game, genome in zip(games, generation):
                score = game.score
                fitness = -score[1] / PLAY_COUNT * 100
                genome["fitness"] = fitness
                play_scores.append(score)

                print(f"Player {game.id} {[layer[0] for layer in genome['brain']]}")
                print(
                    f" - Fitness: {fitness:.1f}% (wins: {score[0]}, losses: {score[1]})"
                )

            print(f"\nGeneration {generation_index:04d} Genomes:")
            best_from_generation = {"fitness": -1e6}
            for i, genome in enumerate(generation):
                print(f"\nPlayer {i}:")
                print_genome(genome)

                if genome["fitness"] > best_from_generation["fitness"]:
                    best_from_generation = genome

            if best_from_generation["fitness"] > best_genome["fitness"]:
                best_genome = best_from_generation
                print("NEW ALL TIME BEST GENOME:")
            else:
                print("All time best genome:")
            print_genome(best_genome)

            losses = np.array([score[1] for score in play_scores])
            brains = [
                [layer[0] for layer in candidate["brain"]] for candidate in generation
            ]
            brain_sizes = np.array([brain_size(layers) for layers in brains])

            losses_min.append(losses.min())
            losses_avg.append(losses.mean())
            losses_max.append(losses.max())

            brain_size_min.append(brain_sizes.min())
            brain_size_avg.append(brain_sizes.mean())
            brain_size_max.append(brain_sizes.max())

            generations = list(range(1, generation_index + 1))

            graph_data = {
                "score_random": (losses_min, losses_avg, losses_max, generations),
                "brain_size": (
                    brain_size_min,
                    brain_size_avg,
                    brain_size_max,
                    generations,
                ),
            }
            plotter.update_data(graph_data)

        except KeyboardInterrupt:
            running = False

    plotter.save_image(f"plots/evolution.png")
