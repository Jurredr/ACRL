import socket


class ACSocket:
    """
    Socket connection with the Assetto Corsa app.
    This is used to get real-time data from the game and send it to the RL model.
    """

    sock = None
    conn = None
    addr = None
    data = None

    def __init__(self, host: str = "127.0.0.1", port: int = 65431) -> None:
        """
        Set up the socket connection.
        :param host: The host to connect to (default: localhost)
        :param port: The port to connect to (default: 65431)
        """

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen()
        print("Socket listening on", (host, port))

    def connect(self) -> socket:
        """
        Wait for an incoming connection and return the socket object.
        """
        self.conn, self.addr = self.sock.accept()
        print("Connected by", self.addr)
        return self.conn

    def update(self) -> None:
        """
        Update the data attribute with the latest data from the socket.
        """
        try:
            self.data = self.conn.recv(1024)
            print("Received data from client")
        except:
            print("No data received from client, closing socket connection")
            self.on_close()

    def on_close(self) -> None:
        """
        Ensure socket is properly closed before terminating program.
        """
        print("Closing socket connection")
        self.sock.close()
