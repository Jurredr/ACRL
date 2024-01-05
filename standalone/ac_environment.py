import numpy as np


class ACEnvironment:
    """
    The environment class for the Assetto Corsa game environment.
    It holds the game state and the observations communicated over the socket.
    """

    observations = None
    next_observations = None

    def __init__(self, data):
        """
        Initialize the environment with the initial data from the socket.

        :param data: The data received from the socket in bytes.
        """
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
        # Convert the byte data to a string
        data_str = data.decode('utf-8')

        # Split the string by commas and map values to a dictionary
        data_dict = dict(map(lambda x: x.split(':'), data_str.split(',')))

        self.track_progress = float(data_dict['track_progress'])
        self.speed_kmh = float(data_dict['speed_kmh'])
        self.world_loc = np.array([float(data_dict['world_loc[0]']), float(
            data_dict['world_loc[1]']), float(data_dict['world_loc[2]'])])
        self.throttle = float(data_dict['throttle'])
        self.brake = float(data_dict['brake'])
        self.steer = float(data_dict['steer'])
        self.lap_time = float(data_dict['lap_time'])
        self.lap_invalid = bool(data_dict['lap_invalid'])
        self.lap_count = int(data_dict['lap_count'])

    def arr(self):
        """
        Get the observations from the game state.

        :return: The observations in a numpy array.
        """
        return np.array([self.track_progress, self.speed_kmh, self.world_loc, self.throttle, self.brake, self.steer, self.lap_time, self.lap_invalid, self.lap_count])
