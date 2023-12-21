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
l_icurrenttime = 0
l_position = 0

# Rounding decimals
DECIMALS = 3


def acMain(ac_version):
    ac.console("[ACRL] Initializing...")

    # Create the app window
    APP_WINDOW = ac.newApp(APP_NAME)
    ac.setSize(APP_WINDOW, 400, 400)
    ac.setTitle(APP_WINDOW, APP_NAME +
                ": Reinforcement Learning")

    # Background
    ac.setBackgroundOpacity(APP_WINDOW, 1)
    ac.setBackgroundColor(APP_WINDOW, 0, 0, 0)

    # Create the labels
    global l_speedkmh, l_laptime, l_normsplinepos, l_velocity, l_worldpos, l_pitch, l_roll, l_disttraveled, l_icurrenttime, l_position

    l_speedkmh = ac.addLabel(APP_WINDOW, "Speed (km/h): 0")
    ac.setPosition(l_speedkmh, 10, 40)

    l_laptime = ac.addLabel(APP_WINDOW, "Lap Time: 0")
    ac.setPosition(l_laptime, 10, 70)

    l_normsplinepos = ac.addLabel(APP_WINDOW, "Normalized Spline Position: 0")
    ac.setPosition(l_normsplinepos, 10, 100)

    l_velocity = ac.addLabel(APP_WINDOW, "Velocity: 0")
    ac.setPosition(l_velocity, 10, 130)

    l_worldpos = ac.addLabel(APP_WINDOW, "World Position: 0")
    ac.setPosition(l_worldpos, 10, 160)

    l_pitch = ac.addLabel(APP_WINDOW, "Pitch: 0")
    ac.setPosition(l_pitch, 10, 190)

    l_roll = ac.addLabel(APP_WINDOW, "Roll: 0")
    ac.setPosition(l_roll, 10, 220)

    l_disttraveled = ac.addLabel(APP_WINDOW, "Distance Traveled: 0")
    ac.setPosition(l_disttraveled, 10, 250)

    l_icurrenttime = ac.addLabel(APP_WINDOW, "iCurrentTime: 0")
    ac.setPosition(l_icurrenttime, 10, 280)

    l_position = ac.addLabel(APP_WINDOW, "position: 0")
    ac.setPosition(l_position, 10, 310)

    ac.console("[ACRL] Initialized")
    return APP_NAME


def acUpdate(deltaT):
    global l_speedkmh, l_laptime, l_normsplinepos, l_velocity, l_worldpos, l_pitch, l_roll, l_disttraveled, l_icurrenttime, l_position

    # Update the labels
    ac.setText(
        l_speedkmh, "Speed (km/h): {}".format(ac.getCarState(0, round(acsys.CS.SpeedKMH, DECIMALS))))
    ac.setText(l_laptime, "Lap Time: {}".format(
        ac.getCarState(0, acsys.CS.LapTime)))
    ac.setText(l_normsplinepos, "Normalized Spline Position: {}".format(
        ac.getCarState(0, round(acsys.CS.NormalizedSplinePosition, DECIMALS))))
    ac.setText(l_velocity, "Velocity: {}".format(
        ac.getCarState(0, acsys.CS.Velocity)))
    ac.setText(l_worldpos, "World Position: {}".format(
        ac.getCarState(0, acsys.CS.WorldPosition)))
    ac.setText(l_pitch, "Pitch: {}".format(
        round(info.physics.pitch, DECIMALS)))
    ac.setText(l_roll, "Roll: {}".format(round(info.physics.roll, DECIMALS)))
    ac.setText(l_disttraveled, "Distance Traveled: {}".format(
        info.graphics.distanceTraveled))
    ac.setText(l_icurrenttime, "iCurrentTime: {}".format(
        info.graphics.iCurrentTime))
    ac.setText(l_position, "position: {}".format(
        info.graphics.position))


def acShutdown():
    ac.console("[ACRL] Shutting down...")
