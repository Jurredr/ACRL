class GameState:
    location = None
    speed_kmh = None
    invalidated = None

    lap_time = None
    lap_count = None

    throttle = None
    brake = None
    steer = None

    def update(self, data):
        """
        Update the game state with the latest data from the socket.

        :param data: The data received from the socket.
        """
        self.location = data["location"]
        self.speed_kmh = data["speed_kmh"]
        self.invalidated = data["invalidated"]

        self.lap_time = data["lap_time"]
        self.lap_count = data["lap_count"]

        self.throttle = data["throttle"]
        self.brake = data["brake"]
        self.steer = data["steer"]
