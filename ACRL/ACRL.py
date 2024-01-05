import sys
import os
import platform
import socket

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

# Model variables
model_running = False

# Socket variables
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server
sock = None

# Label & button variables
label_model_info = None
btn_start = None
btn_stop = None


def acMain(ac_version):
    """
    The main function of the app, called on app start.
    :param ac_version: The version of Assetto Corsa as a string.
    """
    global label_model_info, btn_start, btn_stop
    ac.console("[ACRL] Initializing...")

    # Create the app window
    APP_WINDOW = ac.newApp(APP_NAME)
    ac.setSize(APP_WINDOW, 320, 150)
    ac.setTitle(APP_WINDOW, APP_NAME +
                ": Reinforcement Learning")

    # Background fully black
    ac.setBackgroundOpacity(APP_WINDOW, 1)

    # Info label
    label_model_info = ac.addLabel(
        APP_WINDOW, "Model Running: " + str(model_running) + "\nClick start to begin!")
    ac.setPosition(label_model_info, 320/2, 40)
    ac.setFontAlignment(label_model_info, "center")

    # Start button
    btn_start = ac.addButton(APP_WINDOW, "Start Model")
    ac.setPosition(btn_start, 20, 100)
    ac.setSize(btn_start, 120, 30)
    ac.addOnClickedListener(btn_start, start)
    ac.setVisible(btn_start, 1)

    # Stop button
    btn_stop = ac.addButton(APP_WINDOW, "Stop Model")
    ac.setPosition(btn_stop, 320/2 + 10, 100)
    ac.setSize(btn_stop, 120, 30)
    ac.addOnClickedListener(btn_stop, stop)
    ac.setVisible(btn_stop, 0)

    # Try to connect to socket
    connect()

    ac.console("[ACRL] Initialized")
    return APP_NAME


def acUpdate(deltaT):
    """
    The update function of the app, called every frame.
    Here we get the data from the game, and send it over the socket to the RL model.
    :param deltaT: The time since the last frame as a float.
    """
    global label_model_info, model_running, episode

    # Update the model info label
    ac.setText(label_model_info, "Model Running: " + str(model_running) +
               ("\nEpisode: " + str(episode) if episode > 0 else "\nClick start to begin!"))

    # If the model is not running, don't do anything
    if not model_running:
        return

    # Try to connect to socket if necessary
    connect()

    # Send the data to the model
    try:
        sock.sendall(str.encode(','))
    except:
        ac.console("[ACRL] EXCEPTION: could not send data!")


def acShutdown():
    """
    The shutdown function of the app, called on app close.
    """
    global model_running
    model_running = False
    sock.close()
    ac.console("[ACRL] Shutting down...")


def start(*args):
    """
    The function called when the start button is pressed.
    :param args: The arguments passed to the function.
    """
    global btn_start, btn_stop, episode, model_running, deltaT_total
    ac.console("[ACRL] Starting model...")

    ac.setVisible(btn_start, 0)
    ac.setVisible(btn_stop, 1)
    episode = 1
    deltaT_total = 0
    model_running = True


def stop(*args):
    """
    The function called when the stop button is pressed.
    :param args: The arguments passed to the function.
    """
    global btn_start, btn_stop, model_running, deltaT_total, episode
    ac.console("[ACRL] Stopping model...")

    ac.setVisible(btn_start, 1)
    ac.setVisible(btn_stop, 0)
    model_running = False
    deltaT_total = 0
    episode = -1


def connect():
    """
    Connects to the socket server.
    """
    global sock
    if sock is None:
        try:
            # try connecting to socket server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((HOST, PORT))
            ac.console("[ACRL] Socket connection successful!")
        except:
            ac.console("[ACRL] Socket could not connect to host...")


def respawn():
    """
    Respawns the car at the finish line.
    """
    ac.console("[ACRL] Respawning...")
    # Restart to session menu
    sendCMD(68)
    # Start the lap + driving
    sendCMD(69)
