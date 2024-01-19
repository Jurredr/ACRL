import sys
import os
import platform
import socket
import threading
import ac_api.car_info as ci
import ac_api.input_info as ii
import ac_api.lap_info as li

# The name of the app (ACRL: Assetto Corsa Reinforcement Learning)
APP_NAME = 'ACRL'

# Add the third party libraries to the path
try:
    if platform.architecture()[0] == "64bit":
        sysdir = "stdlib64"
    else:
        sysdir = "stdlib"
    sys.path.insert(
        len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME))
    os.environ['PATH'] += ";."
    sys.path.insert(len(sys.path), os.path.join(
        'apps/python/{}/third_party'.format(APP_NAME), sysdir))
    os.environ['PATH'] += ";."
except Exception as e:
    ac.log("[ACRL] Error importing libraries: %s" % e)

import ac  # noqa: E402
from IS_ACUtil import *  # noqa: E402

# Training enabled flag
training = False
completed = False

# Socket variables
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65431  # The port used by the server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connected = False
# Socket thread
t_sock = None

# Respawn thread
t_res = None
RES_KEY = 121  # F10

# Label & button variables
label_model_info = None
btn_start = None


def acMain(ac_version):
    """
    The main function of the app, called on app start.
    :param ac_version: The version of Assetto Corsa as a string.
    """
    global label_model_info, btn_start, t_res
    ac.console("[ACRL] Initializing...")

    # Create the app window
    APP_WINDOW = ac.newApp(APP_NAME)
    ac.setSize(APP_WINDOW, 320, 140)
    ac.setTitle(APP_WINDOW, APP_NAME +
                ": Reinforcement Learning")

    # Background fully black
    ac.setBackgroundOpacity(APP_WINDOW, 1)

    # Info label
    label_model_info = ac.addLabel(
        APP_WINDOW, "Training: " + str(training) + "\nClick start to begin!")
    ac.setPosition(label_model_info, 320/2, 40)
    ac.setFontAlignment(label_model_info, "center")

    # Start button
    btn_start = ac.addButton(APP_WINDOW, "Start Training")
    ac.setPosition(btn_start, 20, 90)
    ac.setSize(btn_start, 280, 30)
    ac.addOnClickedListener(btn_start, start)
    ac.setVisible(btn_start, 1)

    # Start the respawn listener thread
    t_res = threading.Thread(target=respawn_listener)
    t_res.start()

    ac.console("[ACRL] Initialized")
    return APP_NAME


def acUpdate(deltaT):
    """
    The update function of the app, called every frame.
    :param deltaT: The time since the last frame as a float.
    """
    # Update the model info label
    global completed, training
    if completed:
        ac.setText(label_model_info, "Training completed!" +
                   "\nRestart to train again!")
    else:
        ac.setText(label_model_info, "Training: " + str(training))

    if ac.getCameraMode() is not 4:
        # Lock the camera mode to helicopter
        ac.setCameraMode(4)


def acShutdown():
    """
    The shutdown function of the app, called on app close.
    """
    global training
    training = False
    try:
        stop()
        t_res.join()
        t_sock.join()
    except:
        pass
    ac.console("[ACRL] Shutting down...")


def start(*args):
    """
    The function called when the start button is pressed.
    :param args: The arguments passed to the function.
    """
    global btn_start, training, connected, t_sock
    if not connect():
        ac.console("[ACRL] Didn't start model, could not connect to socket!")
        connected = False
        training = False
        return

    ac.console("[ACRL] Starting model...")

    ac.setVisible(btn_start, 0)
    training = True

    # Start the socket listener thread
    if t_sock is None:
        t_sock = threading.Thread(target=sock_listener)
    t_sock.start()


def stop(*args):
    """
    The function called when the training has stopped.
    :param args: The arguments passed to the function.
    """
    global btn_start, training, sock, connected, t_sock, completed

    ac.console("[ACRL] Stopping model...")
    sock.close()
    connected = False
    training = False
    completed = True


def connect():
    """
    Attempts to connect to the socket server.
    """
    global sock, connected
    if connected:
        return True
    try:
        sock.connect((HOST, PORT))
        connected = True
        ac.console("[ACRL] Socket connection successful!")
        return True
    except:
        ac.console("[ACRL] Socket could not connect to host...")
        return False


def respawn_listener():
    """
    Listens for particular key press and will respawn the car at the finish line when pressed.
    """
    global completed
    while not completed:
        if getKeyState(RES_KEY):
            ac.console("[ACRL] Respawning...")
            # Restart to session menu
            sendCMD(68)
            # Start the lap + driving
            sendCMD(69)


def sock_listener():
    global sock, training
    while True:

        # If not training, don't do anything
        if not training:
            break

        # If the socket is not connected, try to connect
        if not connect():
            ac.console(
                "[ACRL] Socket could not connect to host in acUpdate, stopping training!")
            stop()
            break

        # Receive signals from the socket
        data = sock.recv(1024)
        if not data:
            ac.console("[ACRL] Received stop signal, stopping training...")
            stop()
            break
        ac.console("[ACRL] Received request, responding with game data...")

        # Get the data from the game
        track_progress = ci.get_location()
        speed_kmh = ci.get_speed()
        world_loc = ci.get_world_location()
        throttle = ii.get_gas_input()
        brake = ii.get_brake_input()
        steer = ii.get_steer_input()
        lap_time = li.get_current_lap_time()
        lap_invalid = li.get_invalid()
        lap_count = li.get_lap_count()
        velocity = ci.get_velocity()

        # Turn the data into a string
        data = "track_progress:" + str(track_progress) + "," + "speed_kmh:" + str(speed_kmh) + "," + "world_loc[0]:" + str(world_loc[0]) + "," + "world_loc[1]:" + str(world_loc[1]) + "," + "world_loc[2]:" + str(world_loc[2]) + "," + "throttle:" + str(
            throttle) + "," + "brake:" + str(brake) + "," + "steer:" + str(steer) + "," + "lap_time:" + str(lap_time) + "," + "lap_invalid:" + str(lap_invalid) + "," + "lap_count:" + str(lap_count) + "," + "velocity[0]:" + str(velocity[0]) + "," + "velocity[1]:" + str(velocity[1]) + "," + "velocity[2]:" + str(velocity[2])

        # Send the data in bytes
        sock.sendall(str.encode(data))
