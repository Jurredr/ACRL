import os

import numpy as np
from ac_socket import ACSocket
from gymnasium.wrappers import TimeLimit
from sac.ac_environment import AcEnv
from sac.utils.logx import colorize
from sac.sac import SacAgent


def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    print(colorize("\n--- Assetto Corsa Reinforcement Learning ---\n",
          "magenta", bold=True))
    if input(colorize("Load previous model? (y/n): ", "gray")) == "y":
        load_path = input(
            colorize("Enter model directory (relative): ", "gray"))
        # Check if load_dir exists and if it is a directory and not empty
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                colorize("Directory does not exist!", "red"))
        if not os.path.isdir(load_path):
            raise NotADirectoryError(
                colorize("Path is not a directory!", "red"))
        if not os.listdir(load_path):
            raise ValueError(colorize("Directory is empty!", "red"))

        # The experiment name is the name of the directory
        exp_name = load_path.split("/")[-1]
        print(colorize("Loading model for experiment '" +
              exp_name + "' from " + load_path + "...", "green"))
    else:
        load_path = None

    exp_name = input(colorize("Enter experiment name: ", "gray"))
    print("")

    # Car data (Ferrari 458 GT2)
    max_speed = 270.0
    steer_scale = [-270, 270]

    # Track data (Silverstone 1967)
    spline_points = np.loadtxt(
        '../track_data/spline_points.csv', delimiter=',')

    # Initialize the environment, max_episode_steps is the maximum amount of steps before the episode is truncated
    env = TimeLimit(AcEnv(max_speed=max_speed,
                    steer_scale=steer_scale, spline_points=spline_points), max_episode_steps=1000)

    # Initialize the agent
    hyperparams = {
        "gamma": 0.99,
        "polyak": 0.995,  # 1.0 - tau (soft target update)
        "lr": 1e-3,
        "alpha": 0.2,
        "batch_size": 256,
        "n_episodes": 1000,
        "update_after": 10000,
        "update_every": 50
    }

    agent = SacAgent(env, exp_name, load_path, **hyperparams)

    # Establish a socket connection
    sock = ACSocket()
    with sock.connect() as conn:

        # Set the socket in the environment
        env.unwrapped.set_sock(sock)

        # Run the training loop
        agent.train()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()
