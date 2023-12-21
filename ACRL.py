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

# Value labels
l_speedkmh = 0
l_laptime = 0
l_normsplinepos = 0
l_velocity = 0
l_worldpos = 0
l_pitch = 0
l_roll = 0
l_disttraveled = 0


def acMain(ac_version):
    ac.console("[ACRL] Initializing...")

    # Create the app window
    APP_WINDOW = ac.newApp(APP_NAME)
    ac.setSize(APP_WINDOW, 400, 400)
    ac.setTitle(APP_WINDOW, APP_NAME +
                ": Reinforcement Learning")

    # Background
    ac.setBackgroundOpacity(APP_WINDOW, 1)
    ac.setBackgroundColor(APP_WINDOW, 255, 255, 255)

    # Create the labels
    global l_speedkmh, l_laptime, l_normsplinepos, l_velocity, l_worldpos, l_pitch, l_roll, l_disttraveled
    l_speedkmh = ac.addLabel(APP_WINDOW, "Speed (km/h): 0")
    l_laptime = ac.addLabel(APP_WINDOW, "Lap Time: 0")
    l_normsplinepos = ac.addLabel(APP_WINDOW, "Normalized Spline Position: 0")
    l_velocity = ac.addLabel(APP_WINDOW, "Velocity: 0")
    l_worldpos = ac.addLabel(APP_WINDOW, "World Position: 0")
    l_pitch = ac.addLabel(APP_WINDOW, "Pitch: 0")
    l_roll = ac.addLabel(APP_WINDOW, "Roll: 0")
    l_disttraveled = ac.addLabel(APP_WINDOW, "Distance Traveled: 0")

    ac.console("[ACRL] Initialized")
    return APP_NAME


def acUpdate(deltaT):
    # Update the labels
    ac.setText(
        l_speedkmh, "Speed (km/h): {}".format(ac.getCarState(0, acsys.CS.SpeedKMH)))
    ac.setText(l_laptime, "Lap Time: {}".format(
        ac.getCarState(0, acsys.CS.LapTime)))
    ac.setText(l_normsplinepos, "Normalized Spline Position: {}".format(
        ac.getCarState(0, acsys.CS.NormalizedSplinePosition)))
    ac.setText(l_velocity, "Velocity: {}".format(
        ac.getCarState(0, acsys.CS.Velocity)))
    ac.setText(l_worldpos, "World Position: {}".format(
        ac.getCarState(0, acsys.CS.WorldPosition)))
    ac.setText(l_pitch, "Pitch: {}".format(info.physics.pitch))
    ac.setText(l_roll, "Roll: {}".format(info.physics.roll))
    ac.setText(l_disttraveled, "Distance Traveled: {}".format(
        info.physics.distanceTraveled))


def acShutdown():
    ac.console("[ACRL] Shutting down...")
