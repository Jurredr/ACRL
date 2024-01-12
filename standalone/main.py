import numpy as np
from ac_socket import ACSocket
from sac.sac import Agent
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
from sac.ac_environment import AcEnv


def main():
    """
    The main function of the standalone application.
    It connects to the Assetto Corsa app, gets data, runs it through the RL model, and sends actions back to the game.
    """
    # Car data (ferrari 312t)
    max_speed = 300.0
    steer_scale = [-220, 220]

    # Initialize the environment, max_episode_steps is the maximum amount of steps before the episode is truncated
    env = TimeLimit(AcEnv(max_speed, steer_scale), max_episode_steps=300)

    # Initialize the agent
    agent = Agent(input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0])

    # Establish a socket connection
    sock = ACSocket()

    # Scores and amount of episodes to run
    best_score = env.reward_range[0]
    score_history = []
    n_episodes = 500

    # Loop until the socket is closed or the program is terminated
    with sock.connect() as conn:
        for i in range(n_episodes):
            observation = env.reset(sock=sock)
            done = False
            score = 0

            while not done:
                action = agent.choose_action(observation)
                observation_, reward, terminated, truncated, info = env.step(
                    sock=sock, action=action)
                score += reward
                done = terminated or truncated
                agent.remember(observation, action, reward,
                               observation_, done)
                agent.learn()
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        sock.end_training()
        sock.on_close()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()
