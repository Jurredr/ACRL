import numpy as np


class ACEnvironment:
    """
    The environment class for the Assetto Corsa game environment.
    It holds the game state and the observations communicated over the socket.
    """

    observations = None
    next_observations = None

    def __init__(self, data, max_speed=200.0, lap_completion_bonus=1.0, off_track_penalty=-1.0):
        """
        Initialize the environment with the initial data from the socket.
        :param data: The data received from the socket in bytes.
        :param max_speed: The maximum speed of the car in km/h.
        :param lap_completion_bonus: The bonus reward for completing a lap.
        :param off_track_penalty: The penalty for going off track.
        """
        self.max_speed = max_speed
        self.lap_completion_bonus = lap_completion_bonus
        self.off_track_penalty = off_track_penalty
        self.observations = self.update(data)

    def update(self, data, next=False):
        """
        Update the game state with the latest data from the socket.
        :param data: The data received from the socket in bytes.
        :param next: Whether the next_observations should be updated.
        """
        obs = ACObservation(data)

        if next:
            self.next_observations = obs
        else:
            self.observations = obs

    def progress(self):
        """
        Progress the environment to the next state.
        """
        self.observations = self.next_observations
        self.next_observations = None

    def get_reward(self):
        """
        Get the reward from the current environment observations.
        :return: The reward.
        """
        # Reward for how far the car has come on the track [0.0, 1.0]
        progress_reward = self.next_observations.track_progress

        # Give a reward for the current speed, going faster is better for the lap time
        speed_reward = self.next_observations.speed_kmh / \
            self.max_speed  # Normalize speed to [0.0, 1.0]

        # Penalty for Going Off Track (default: -1.0)
        off_track_penalty = self.off_track_penalty if self.next_observations.lap_invalid else 0.0

        # Lap completion bonus; an extra bonus for actually reaching the finish line (default: 1.0)
        lap_completion_reward = self.lap_completion_bonus if self.next_observations.lap_count > 0 else 0.0

        # Combine individual rewards
        total_reward = lap_completion_reward + \
            progress_reward + speed_reward + off_track_penalty

        # Minimum reward is -1.0, maximum reward is 3.0
        return total_reward

    def episode_done(self, timeout=60*1000):
        """
        Check whether an episode is finished with the current environment states.
        :param timeout: The timeout in ms after which an episode has taken too long (it took too long to reach the finish line).
        :return: Whether the episode is done.
        """
        # TODO: Ideally, we would want to check whether the car is stuck, e.g. through checking if speed is too low (or zero) for too long
        return self.observations.lap_invalid or self.observations.lap_count > 0 or self.observations.track_progress == 1.0 or self.observations.lap_time >= timeout


class ACObservation:
    """
    The observation class holding different data states from the game.
    """

    def __init__(self, data):
        """
        Initialize the environment with observation data.
        :param data: The data in bytes.
        """
        self.update(data)

    def update(self, data):
        """
        Update the observation data states from bytes to the correct types.
        :param data: The data in bytes.
        """
        try:
            # Convert the byte data to a string
            data_str = data.decode('utf-8')

            # Split the string by commas and map values to a dictionary
            data_dict = dict(map(lambda x: x.split(':'), data_str.split(',')))
        except:
            # If the data is invalid, return
            return

        self.track_progress = float(data_dict['track_progress'])
        self.speed_kmh = float(data_dict['speed_kmh'])
        self.world_loc = np.array([float(data_dict['world_loc[0]']), float(
            data_dict['world_loc[1]']), float(data_dict['world_loc[2]'])])
        self.throttle = float(data_dict['throttle'])
        self.brake = float(data_dict['brake'])
        self.steer = float(data_dict['steer'])
        self.lap_time = float(data_dict['lap_time'])
        # Lap stays invalid as soon as it has been invalid once
        self.lap_invalid = bool(data_dict['lap_invalid']) or (
            self.lap_invalid is not None and self.lap_invalid)
        self.lap_count = int(data_dict['lap_count'])

    def arr(self):
        """
        Get the observations from the game state.
        :return: The observations in a numpy array.
        """
        return np.array([self.track_progress, self.speed_kmh, self.world_loc, self.throttle, self.brake, self.steer, self.lap_time, self.lap_invalid, self.lap_count])
