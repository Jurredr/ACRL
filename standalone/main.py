import numpy as np
from ac_controller import ACController
from ac_socket import ACSocket
from ac_environment import ACEnvironment


def main():
    """
    The main function of the standalone application.
    It connects to the Assetto Corsa app, gets data, sends it to the model, and sends the output actions back to the game.
    """
    # Scores and amount of episodes to run
    n_episodes = 3

    # 6 seconds timeout for each episode
    TIMEOUT = 6*1000

    # Initialize the controller
    controller = ACController()

    # Establish a socket connection
    sock = ACSocket()

    # Loop until the socket is closed or the program is terminated
    with sock.connect() as conn:

        # Loop episodes
        for i in range(n_episodes):
            step = 0
            score = 0

            # 1. Get the initial observation from the game and initialize the environment
            sock.update()
            env = ACEnvironment(sock.data)

            # 2. Loop actions-observations until the episode is done
            while not env.episode_done(timeout=TIMEOUT):
                try:
                    # 4. Perform the action in the game
                    controller.perform(1.0, 0.0, 0.0)
                    step += 1
                except:
                    sock.on_close()
                    break

            # 9. Reset the car back to the start
            controller.reset_car()

            print("[ACRL] Episode {} finished after {} steps with score {}".format(
                i, step, score))


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()
