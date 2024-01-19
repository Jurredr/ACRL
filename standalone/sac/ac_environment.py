import math
from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from ac_controller import ACController
from ac_socket import ACSocket
from sac.utils.logx import colorize
from sac.utils.track_spline import get_heading_error, get_distance_to_center_line


class AcEnv(gym.Env):
    """
    The custom gymnasium environment for the Assetto Corsa game.
    """

    metadata = {"render_modes": [], "render_fps": 0}
    _observations = None
    _invalid_flag = 0.0
    _sock = None

    def __init__(self, render_mode: Optional[str] = None, max_speed=200.0, steer_scale=[-360, 360], spline_points=[[], []]):
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
        # - "previous_track_progress": The previous track progress [0.0, 1.0]
        # - "velocity_x": The x velocity of the car
        # - "velocity_z": The z velocity of the car
        self.observation_space = spaces.Box(
            low=np.array(
                [0.000, 0.0, -2000.0, -2000.0, -2000.0, 0.0, 1.0, 0.000, -2000.0, -2000.0]),
            high=np.array([1.000, max_speed, 2000.0,
                          2000.0, 2000.0, 1.0, 2.0, 1.000, 2000.0, 2000.0]),
            shape=(10,),
            dtype=np.float32,
        )

        # We have a continuous action space, where we have:
        # - A brake/throttle value, which is a number in [-1.0, 1.0]
        # - A steering angle, which is a number in [-1.000, 1.000]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.000]),
            high=np.array([1.0, 1.000]),
            shape=(2,),
            dtype=np.float32
        )

        # Set the spline points for the track center line
        self.spline_points = spline_points

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
            print(colorize("Error parsing data, returning empty dict!", "red"))
            return {}

        # throttle = float(data_dict['throttle'])
        # brake = float(data_dict['brake'])
        # steer = float(data_dict['steer'])
        # lap_time = float(data_dict['lap_time'])

        # print(data_dict)
        track_progress = float(data_dict['track_progress'])
        speed_kmh = float(data_dict['speed_kmh'])
        world_loc_x = float(data_dict['world_loc[0]'])
        world_loc_y = float(data_dict['world_loc[1]'])
        world_loc_z = float(data_dict['world_loc[2]'])
        lap_count = float(data_dict['lap_count'])
        previous_track_progress = self._observations[0] if self._observations is not None else 0.000
        velocity_x = float(data_dict['velocity[0]'])
        velocity_z = float(data_dict['velocity[1]'])

        # Lap stays invalid as soon as it has been invalid once
        lap_invalid = self._invalid_flag
        if data_dict['lap_invalid'] == 'True':
            lap_invalid = 1.0
        self._invalid_flag = lap_invalid

        # Update the observations
        self._observations = np.array(
            [track_progress, speed_kmh, world_loc_x, world_loc_y, world_loc_z, lap_invalid, lap_count, previous_track_progress, velocity_x, velocity_z], dtype=np.float32)
        return self._observations

    def _get_info(self):
        """
        Extra information returned by step and reset functions.
        """
        return {}

    def _get_reward_1(self, penalty_offtrack=-1.0, penalty_lowspeed=-0.4, min_speed=5.0, min_progress=0.0001, penalty_lowprogress=-0.8, progress_weights=[]):
        """
        A reward considering just speed.
        """
        speed_reward = self._observations[1] / self.max_speed

        # If the car is off track, return a penalty
        if self._observations[5] == 1.0:
            return penalty_offtrack

        # If the car is making too little progress, return a penalty
        if self._observations[0] <= self._observations[7] + min_progress:
            return penalty_lowprogress

        # If the car is moving too slow, return a penalty
        if self._observations[1] <= min_speed:
            return penalty_lowspeed

        # If the car is past a certain progress point, set a weight for the reward
        weight = 1.0
        progress_point = int(self._observations[0] * 10)
        if progress_point >= 1 and progress_point <= 9:
            weight = progress_weights[progress_point - 1]

        return speed_reward * weight

    def _get_reward_2(self, penalty_offtrack=-1.0, penalty_lowspeed=-0.4, min_speed=5.0, min_progress=0.0001, penalty_lowprogress=-0.8, progress_weights=[]):
        """
        A reward considering speed and progress on track.
        """
        speed_reward = self._observations[1] / self.max_speed
        progress_reward = self._observations[0]

        # If the car is off track, return a penalty
        if self._observations[5] == 1.0:
            return penalty_offtrack

        # If the car is making too little progress, return a penalty
        if self._observations[0] <= self._observations[7] + min_progress:
            return penalty_lowprogress

        # If the car is moving too slow, return a penalty
        if self._observations[1] <= min_speed:
            return penalty_lowspeed

        # If the car is past a certain progress point, set a weight for the reward
        weight = 1.0
        progress_point = int(self._observations[0] * 10)
        if progress_point >= 1 and progress_point <= 9:
            weight = progress_weights[progress_point - 1]

        return (progress_reward + speed_reward) * weight

    def _get_reward_3(self, penalty_offtrack=-1.0, penalty_lowspeed=-0.4, min_speed=5.0, min_progress=0.0001, penalty_lowprogress=-0.8, progress_weights=[]):
        """
        A reward just considering delta progress on track.
        """
        progress = self._observations[0]
        previous_progress = self._observations[7]
        delta_progress = progress - previous_progress

        # If the car is off track, return a penalty
        if self._observations[5] == 1.0:
            return penalty_offtrack

        # If the car is making too little progress, return a penalty
        if self._observations[0] <= self._observations[7] + min_progress:
            return penalty_lowprogress

        # If the car is moving too slow, return a penalty
        if self._observations[1] <= min_speed:
            return penalty_lowspeed

        # If the car is past a certain progress point, set a weight for the reward
        weight = 1.0
        progress_point = int(self._observations[0] * 10)
        if progress_point >= 1 and progress_point <= 9:
            weight = progress_weights[progress_point - 1]

        return delta_progress * weight

    def _get_reward_4(self, penalty_offtrack=-1.0, penalty_lowspeed=-0.4, min_speed=5.0, min_progress=0.0001, penalty_lowprogress=-0.8, progress_weights=[]):
        """
        A reward considering speed and delta progress on track.
        """
        speed_reward = self._observations[1] / self.max_speed
        progress = self._observations[0]
        previous_progress = self._observations[7]
        delta_progress = progress - previous_progress

        # If the car is off track, return a penalty
        if self._observations[5] == 1.0:
            return penalty_offtrack

        # If the car is making too little progress, return a penalty
        if self._observations[0] <= self._observations[7] + min_progress:
            return penalty_lowprogress

        # If the car is moving too slow, return a penalty
        if self._observations[1] <= min_speed:
            return penalty_lowspeed

        # If the car is past a certain progress point, set a weight for the reward
        weight = 1.0
        progress_point = int(self._observations[0] * 10)
        if progress_point >= 1 and progress_point <= 9:
            weight = progress_weights[progress_point - 1]

        return (delta_progress + speed_reward) * weight

    def _get_reward_5(self, weight_wrongdir=1.0, weight_offcenter=1.0, weight_extra_offcenter=1.0, extra_offcenter_penalty=False):
        """
        A reward considering speed, angle and distance from center of track.
        :return: The reward.
        """
        speed = self._observations[1]  # speed in the forward direction
        world_x = self._observations[2]
        world_z = self._observations[4]
        velocity_x = self._observations[8]
        velocity_z = self._observations[9]
        # angle in degrees between car direction and track direction
        theta = get_heading_error(self.spline_points, world_x, world_z, [
                                  velocity_x, velocity_z])
        dist_offcenter = get_distance_to_center_line(
            self.spline_points, world_x, world_z)  # distance from center of track

        reward = (speed * math.cos(theta)) - (weight_wrongdir *
                                              speed * math.sin(theta)) - (weight_offcenter * speed * abs(dist_offcenter))
        if extra_offcenter_penalty:
            reward -= (weight_extra_offcenter * abs(dist_offcenter))

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
        self._invalid_flag = 0.0
        observation = self._update_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray, ignore_done: bool = False):
        """
        Perform an action in the environment and get the results.
        :param action: The action to perform
        :return: The observation, reward, terminated, truncated, info
        """
        # Perform the action in the game
        # print("action", action)
        self.controller.perform(action[0], action[1])

        # Get the new observations
        observation = self._update_obs()

        # Progress goal (100% of the track, with 0.99 to account for errors)
        progress_goal = 0.99

        lap_invalid = observation[5]
        lap_count = observation[6]
        track_progress = observation[0]
        if ignore_done:
            terminated = False
        else:
            delta_prog = observation[0] - observation[7]
            terminated = lap_invalid == 1.0 or delta_prog <= 0.00005 or lap_count > 1.0 or track_progress >= progress_goal

        # Truncated gets updated based on timesteps by TimeLimit wrapper
        truncated = False

        # Get the reward and info
        reward = self._get_reward_5()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment; a PyGame renderer is not needed for AC.
        """
        print(colorize("Rendering not supported for this environment!", "red"))

    def close(self):
        """
        Close the environment and the socket connection.
        """
        self._sock.end_training()
        self._sock.on_close()
