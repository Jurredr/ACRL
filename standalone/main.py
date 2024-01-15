import numpy as np
from ac_socket import ACSocket
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
from sac.ac_environment import AcEnv
from sac.sac import sac


def main():
    """
    The main function of the standalone application.
    It connects to the Assetto Corsa app, gets data, runs it through the RL model, and sends actions back to the game.
    """
    # Car data (Ferrari 458 GT2)
    max_speed = 270.0
    steer_scale = [-270, 270]

    # Initialize the environment, max_episode_steps is the maximum amount of steps before the episode is truncated
    def env(): return TimeLimit(AcEnv(max_speed=max_speed,
                                      steer_scale=steer_scale), max_episode_steps=300)

    # Initialize the agent
    # agent = Agent(input_dims=env.observation_space.shape,
    #               env=env, n_actions=env.action_space.shape[0])
    agent = sac(env, epochs=5, steps_per_epoch=100)

    # # Establish a socket connection
    # sock = ACSocket()

    # # Scores and amount of episodes to run
    # best_score = env.reward_range[0]
    # score_history = []
    # n_episodes = 500

    # # Loop until the socket is closed or the program is terminated
    # with sock.connect() as conn:
    #     print("Starting training...")
    #     env.unwrapped.set_sock(sock)

    #     for i in range(n_episodes):
    #         print("--- Starting episode:", i + 1, "/", n_episodes)
    #         observation, _ = env.reset()
    #         done = False
    #         score = 0

    #         while not done:
    #             action = agent.choose_action(observation)
    #             observation_, reward, terminated, truncated, info = env.step(
    #                 action=action)
    #             done = terminated or truncated
    #             score += reward
    #             agent.remember(observation, action, reward,
    #                            observation_, done)
    #             agent.learn()
    #             observation = observation_

    #         print("=== Finished episode", i + 1,
    #               "/", n_episodes, "[score]:", score)
    #         score_history.append(score)
    #         avg_score = np.mean(score_history[-100:])

    #         if avg_score > best_score:
    #             best_score = avg_score
    #             agent.save_models()

    #     print("Training completed! Best score:",
    #           best_score, "Average score:", avg_score)
    #     sock.end_training()
    #     sock.on_close()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()
