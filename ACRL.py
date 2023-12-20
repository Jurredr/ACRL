import sys
import os
import platform

# The name of the app
APP_NAME = 'ACRL'

# Add the third party libraries to the path
if platform.architecture()[0] == "64bit":
    sysdir = "stdlib64"
else:
    sysdir = "stdlib"
sys.path.insert(len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME))
os.environ['PATH'] += ";."
sys.path.insert(len(sys.path), os.path.join(
    'apps/python/{}/third_party'.format(APP_NAME), sysdir))
os.environ['PATH'] += ";."

# Import the Assetto Corsa libraries
import ac  # noqa: E402
import acsys  # noqa: E402
from sim_info import info  # noqa: E402

l_lapcount = 0
lapcount = 0


def acMain(ac_version):
    ac.console("[ACRL] Initializing...")

    # Create the app window
    APP_WINDOW = ac.newApp(APP_NAME)
    ac.setSize(APP_WINDOW, 400, 400)
    ac.setTitle(APP_WINDOW, APP_NAME +
                ": Reinforcement Learning")

    global l_lapcount
    l_lapcount = ac.addLabel(APP_WINDOW, "Laps: 0")
    ac.setPosition(l_lapcount, 3, 30)

    ac.console("[ACRL] Initialized")
    return APP_NAME


def acUpdate(deltaT):
    global l_lapcount, lapcount
    ac.console(str(deltaT))
    laps = ac.getCarState(0, acsys.CS.LapCount)

    if laps > lapcount:
        lapcount = laps
        ac.setText(l_lapcount, "Laps: {}".format(lapcount))


def acShutdown():
    ac.console("[ACRL] Shutting down...")
