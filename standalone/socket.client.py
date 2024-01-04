import socket
import time
import struct

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b"Hello, world")
    data = s.recv(1024)

    # Convert the bytes back to a float
    data = struct.unpack('!d', data)[0]

current_time = time.time()
print("It took", (current_time - data) * 1000,
      "ms to get a response from the server.")
