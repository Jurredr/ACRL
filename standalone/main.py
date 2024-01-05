from standalone.ac_socket import ACSocket
from standalone.game_state import GameState


def main():
    """
    The main function of the standalone application.
    It connects to the Assetto Corsa app, gets data, sends it to the model, and sends the output actions back to the game.
    """
    # 1. Initialize the game_state object
    game_state = GameState()

    # TODO: 2. Initialize the model
    # env = gym.make('InvertedPendulumBulletEnv-v0')
    # agent = Agent(input_dims=env.observation_space.shape, env=env,
    #         n_actions=env.action_space.shape[0])

    # 3. Establish socket connection
    sock = ACSocket()

    # 4 Loop until the socket is closed or the program is terminated
    with sock.connect() as conn:
        while True:
            try:
                # 5. Get data from the socket
                sock.update()

                # 6. Update the game_state object with the new data
                game_state.update(sock.data)

                # TODO: 7. Get action from the model
                # action = agent.choose_action(observation)
                # observation_, reward, done, info = env.step(action)
                # agent.remember(observation, action, reward, observation_, done)
                # agent.learn()

                # TODO: 9. Send the model's output to the controller
                # controller.send(action)
            except:
                sock.on_close()
                # TODO: save & close the model
                # agent.save_models()
                break


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()
