import socket
import time
import struct

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break

            current_time = time.time()
            data = struct.unpack('!d', data)[0]
            print("It took", (current_time * 1000 - data * 1000),
                  "ms for the client message to arrive at the server.")

            data = struct.pack('!d', current_time)
            conn.sendall(data)
