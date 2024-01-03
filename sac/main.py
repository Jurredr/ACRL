from collections import UserDict
import gym
import numpy as np
from sac import Agent
import numpy as np
import matplotlib.pyplot as plt

import gym.envs.registration

# Do this before importing pybullet_envs (adds an extra property env_specs as a property to the registry, so it looks like the <0.26 envspec version)
registry = UserDict(gym.envs.registration.registry)
registry.env_specs = gym.envs.registration.registry
gym.envs.registration.registry = registry

import pybullet_envs  # noqa F401


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = Agent(input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0])
    n_games = 250
    filename = 'inverted_pendulum.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    # Whether to load a pretrained model (we are just evaluating)
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    # Loop through the number of games
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        # Loop until the episode is done
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' %
              score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
