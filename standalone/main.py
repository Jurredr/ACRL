import numpy as np
from ac_controller import ACController
from ac_socket import ACSocket
from ac_environment import ACEnvironment
from sac.sac import Agent


def main():
    """
    The main function of the standalone application.
    It connects to the Assetto Corsa app, gets data, sends it to the model, and sends the output actions back to the game.
    """
    # Scores and amount of episodes to run
    best_score = -1.0  # TODO; what should this start as? env.reward_range[0]
    score_history = []
    n_episodes = 500

    # 100 seconds timeout for each episode
    TIMEOUT = 100*1000

    # TODO; Initialize the agent properly
    # https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/main_sac.py
    agent = Agent(alpha=0.003, beta=0.0003, input_dims=[
                  8], max_action=10, gamma=0.99, n_actions=3, max_size=1000000, tau=0.005, batch_size=256, reward_scale=2)

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
            episode_done = False

            # 1. Get the initial observation from the game and initialize the environment
            sock.update()
            env = ACEnvironment(sock.data)

            # 2. Loop actions-observations until the episode is done
            while not episode_done:
                try:
                    # 3. Get action from the model
                    action = agent.choose_action(env.observations.arr())

                    # 4. Perform the action in the game
                    controller.perform(action)

                    # 5. Get the new observation from the game
                    sock.update()
                    env.update(sock.data, next=True)

                    # 6. Get the reward and whether the episode is done
                    reward = env.get_reward()
                    episode_done = env.episode_done(timeout=TIMEOUT)

                    # 7. Save the data to the memory and learn from it
                    agent.remember(env.observations.arr(), action,
                                   reward, env.next_observations.arr(), episode_done)
                    agent.learn()

                    # 8. Update the observation and scores
                    env.progress()
                    score += reward
                    score_history.append(score)
                    avg_score = np.mean(score_history[-100:])
                    if avg_score > best_score:
                        best_score = avg_score
                        agent.save_models()

                    print('[ACRL] Episode ', i, ' step ', step, 'score %.1f' %
                          score, 'avg_score %.1f' % avg_score)
                    step += 1
                except:
                    sock.on_close()
                    agent.save_models()
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
