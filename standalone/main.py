import numpy as np
from ac_socket import ACSocket
from gymnasium.wrappers import TimeLimit
from sac.ac_environment import AcEnv
from sac.sac import SacAgent


def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    # Car data (Ferrari 458 GT2)
    max_speed = 270.0
    steer_scale = [-270, 270]

    # Initialize the environment, max_episode_steps is the maximum amount of steps before the episode is truncated
    env = TimeLimit(AcEnv(max_speed=max_speed,
                    steer_scale=steer_scale), max_episode_steps=300)

    print("--- Assetto Corsa Reinforcement Learning ---")
    exp_name = input("Enter experiment name: ")

    # Initialize the agent
    agent = SacAgent(env, exp_name, n_epochs=2,
                     update_after=100, update_every=10)

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
