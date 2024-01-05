import json


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
        # Convert the byte data to a string
        data_str = data.decode('utf-8')

        # Parse the string as a JSON object
        data_dict = json.loads(data_str)

        self.observations = self.update(data_dict)

    def update(self, data, next=False):
        """
        Update the game state with the latest data from the socket.

        :param data: The data received from the socket in bytes.
        :param next: Whether the next_observations should be updated.
        """
        # Convert the byte data to a string
        data_str = data.decode('utf-8')

        # Parse the string as a JSON object
        data_dict = json.loads(data_str)

        if next:
            self.next_observations = data_dict
        else:
            self.observations = data_dict

    def progress(self):
        """
        Progress the environment to the next state.
        """
        self.observations = self.next_observations
        self.next_observations = None
