import numpy as np

import gymnasium as gym
from gymnasium import spaces

from standalone.ac_controller import ACController


class AcEnv(gym.Env):
    """
    The custom gymnasium environment for the Assetto Corsa game.
    """

    metadata = {"render_modes": [], "render_fps": 0}
    _observations = None

    def __init__(self, render_mode=None, max_speed=200.0, steer_scale=[-360, 360]):
        # Initialize the controller
        self.controller = ACController(steer_scale)
        self.max_speed = max_speed

        # Observations is a dictionary with the following keys:
        # - "track_progress": The progress of the car on the track, in [0.0, 1.0]
        # - "speed_kmh": The speed of the car in km/h
        # - "world_loc": The world location of the car [x, y, z]
        # - "lap_invalid": Whether the current lap is invalid (boolean)
        # - "lap_count": The current lap count (0 or 1)
        # - "steer": The previous steering angle of the car in [-1.000, 1.000]
        self.observation_space = spaces.Dict(
            {
                "track_progress": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "speed_kmh": spaces.Box(0.0, max_speed, shape=(1,), dtype=np.float32),
                "world_loc": spaces.Box(-2000.0, 2000.0, shape=(3,), dtype=np.float32),
                "lap_invalid": spaces.Discrete(2),
                "lap_count": spaces.Discrete(2),
                # "steer": spaces.Box(-1.000, 1.000, shape=(1,), dtype=np.float32),
            }
        )

        # We have a continuous action space, where we have:
        # - A throttle, which is a number in [0.0, 1.0]
        # - A brake, which is a number in [0.0, 1.0]
        # - A steering angle, which is a number in [-1.000, 1.000]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.000]), high=np.array([1.0, 1.0, 1.000]), dtype=np.float32
        )

        # Assert that the render mode is valid
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _update_obs(self, sock):
        """
        Get the current observation from the game over socket.
        """
        # Send a request to the game
        sock.send(b"data_request")

        # Receive the data from the game
        try:
            data = sock.recv(1024)

            # Convert the byte data to a string
            data_str = data.decode('utf-8')

            # Split the string by commas and map values to a dictionary
            data_dict = dict(map(lambda x: x.split(':'), data_str.split(',')))
        except:
            # If the data is invalid, throw error and return empty dict
            print("Error parsing data, returning empty dict!")
            return {}

        # throttle = float(data_dict['throttle'])
        # brake = float(data_dict['brake'])
        # steer = float(data_dict['steer'])
        # lap_time = float(data_dict['lap_time'])

        track_progress = float(data_dict['track_progress'])
        speed_kmh = float(data_dict['speed_kmh'])
        world_loc = np.array([float(data_dict['world_loc[0]']), float(
            data_dict['world_loc[1]']), float(data_dict['world_loc[2]'])])

        # Lap stays invalid as soon as it has been invalid once
        lap_invalid = bool(data_dict['lap_invalid']) or (
            self._observations is not None and self._observations['lap_invalid'] is not None and self._observations['lap_invalid'])
        lap_count = int(data_dict['lap_count'])

        # Update the observations
        self._observations = {
            "track_progress": track_progress,
            "speed_kmh": speed_kmh,
            "world_loc": world_loc,
            "lap_invalid": lap_invalid,
            "lap_count": lap_count,
        }
        return self._observations

    def _get_info(self):
        """
        Extra information returned by step and reset functions.
        """
        return {}

    def _get_reward(self):
        """
        Get the reward from the current environment observations.
        :return: The reward.
        """
        off_track_penalty = -1.0
        lap_completion_bonus = 1.0
        observations = self._observations

        # Reward for how far the car has come on the track [0.0, 1.0]
        progress_reward = observations['track_progress']

        # Give a reward for the current speed, going faster is better for the lap time
        speed_reward = observations['speed_kmh'] / \
            self.max_speed  # Normalize speed to [0.0, 1.0]

        # Penalty for Going Off Track (-1.0)
        off_track_penalty = off_track_penalty if observations['lap_invalid'] else 0.0

        # Lap completion bonus; an extra bonus for actually reaching the finish line (default: 1.0)
        lap_completion_reward = lap_completion_bonus if observations['lap_count'] > 0 else 0.0

        # Combine individual rewards
        total_reward = lap_completion_reward + \
            progress_reward + speed_reward + off_track_penalty

        # Minimum reward is -1.0, maximum reward is 3.0
        return total_reward

    def reset(self, sock, seed=None, options=None):
        """
        Reset the environment to initiate a new episode.
        :param seed: The seed for the environment's random number generator
        :param options: The options for the environment
        :return: The initial observation and info
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the controller
        self.controller.reset_car()

        # Get the initial observations from the game
        observation = self._update_obs(sock)
        info = self._get_info()

        return observation, info

    def step(self, sock, action):
        """
        Perform an action in the environment and get the results.
        :param action: The action to perform
        :return: The observation, reward, terminated, truncated, info
        """
        # Perform the action in the game
        self.controller.perform(action)

        # Get the new observations
        observation = self._update_obs(sock)

        # TODO: add check if speed is too low for a while
        terminated = observation['lap_invalid'] or observation['lap_count'] > 0 or observation['track_progress'] == 1.0

        # Truncated gets updated based on timesteps by TimeLimit wrapper
        truncated = False

        # Get the reward and info
        reward = self._get_reward()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment; a PyGame renderer is not needed for AC.
        """
        print("Rendering not supported for this environment!")

    def close(self):
        """
        Nothing to close.
        """
        print("Closing environment! (nothing to close)")
