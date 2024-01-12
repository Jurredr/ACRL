from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from ac_controller import ACController
from ac_socket import ACSocket


class AcEnv(gym.Env):
    """
    The custom gymnasium environment for the Assetto Corsa game.
    """

    metadata = {"render_modes": [], "render_fps": 0}
    _observations = None
    invalid_flag = 0.0
    _sock = None

    def __init__(self, render_mode: Optional[str] = None, max_speed=200.0, steer_scale=[-360, 360]):
        # Initialize the controller
        self.controller = ACController(steer_scale)
        self.max_speed = max_speed

        # Observations is a Box with the following data:
        # - "track_progress": The progress of the car on the track, in [0.0, 1.0]
        # - "speed_kmh": The speed of the car in km/h [0.0, max_speed]
        # - "world_loc_x": The world's x location of the car [-2000.0, 2000.0]
        # - "world_loc_y": The world's y location of the car [-2000.0, 2000.0]
        # - "world_loc_z": The world's z location of the car [-2000.0, 2000.0]
        # - "lap_invalid": Whether the current lap is valid [0.0, 1.0]
        # - "lap_count": The current lap count [1.0, 2.0]
        self.observation_space = spaces.Box(
            low=np.array([0.000, 0.0, -2000.0, -2000.0, -2000.0, 0.0, 1.0]),
            high=np.array([1.000, max_speed, 2000.0, 2000.0, 2000.0, 1.0, 2.0]),
            shape=(7,),
            dtype=np.float32,
        )

        # We have a continuous action space, where we have:
        # - A throttle, which is a number in [0.0, 1.0]
        # - A brake, which is a number in [0.0, 1.0]
        # - A steering angle, which is a number in [-1.000, 1.000]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.000]),
            high=np.array([1.0, 1.0, 1.000]),
            shape=(3,),
            dtype=np.float32
        )

        # Assert that the render mode is valid
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def set_sock(self, sock: ACSocket):
        self._sock = sock

    def _update_obs(self):
        """
        Get the current observation from the game over socket.
        """
        # Send a request to the game
        self._sock.update()
        data = self._sock.data

        try:
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

        print(data_dict)
        track_progress = float(data_dict['track_progress']), 1
        speed_kmh = float(data_dict['speed_kmh']), 1
        world_loc_x = float(data_dict['world_loc[0]']), 1
        world_loc_y = float(data_dict['world_loc[1]']), 1
        world_loc_z = float(data_dict['world_loc[2]']), 1

        # Lap stays invalid as soon as it has been invalid once
        lap_invalid = self.invalid_flag
        if data_dict['lap_invalid']:
            lap_invalid = 1.0
        self.invalid_flag = lap_invalid
        lap_count = float(data_dict['lap_count'])

        # Update the observations
        self._observations = np.array(
            [track_progress, speed_kmh, world_loc_x, world_loc_y, world_loc_z, 0.0, lap_count], dtype=np.float32)
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
        progress_reward = observations[0]

        # Give a reward for the current speed, going faster is better for the lap time
        speed_reward = observations[1] / \
            self.max_speed  # Normalize speed to [0.0, 1.0]

        # Penalty for Going Off Track (-1.0)
        off_track_penalty = off_track_penalty if observations[5] else 0.0

        # Lap completion bonus; an extra bonus for actually reaching the finish line (default: 1.0)
        lap_completion_reward = lap_completion_bonus if observations[6] > 1.0 else 0.0

        # Combine individual rewards
        total_reward = lap_completion_reward + \
            progress_reward + speed_reward + off_track_penalty

        # Minimum reward is -1.0, maximum reward is 3.0
        return total_reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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
        observation = self._update_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray):
        """
        Perform an action in the environment and get the results.
        :param action: The action to perform
        :return: The observation, reward, terminated, truncated, info
        """
        # Perform the action in the game
        print("action", action)
        self.controller.perform(action[0], action[1], action[2])

        # Get the new observations
        observation = self._update_obs()

        # TODO: add check if speed is too low for a while
        lap_invalid = observation[5]
        lap_count = observation[6]
        track_progress = observation[0]
        # terminated = lap_invalid == 1.0 or lap_count > 1.0 or track_progress == 1.0
        terminated = False

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
