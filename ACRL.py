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
l_velocityX = 0
l_velocityY = 0
l_velocityZ = 0
l_worldposX = 0
l_worldposY = 0
l_worldposZ = 0
l_disttraveled = 0

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
    ac.setBackgroundColor(APP_WINDOW, 255, 255, 255)

    # Create the labels
    global l_speedkmh, l_laptime, l_normsplinepos, l_velocity, l_worldpos, l_disttraveled

    l_speedkmh = ac.addLabel(APP_WINDOW, "Speed (km/h): 0")
    ac.setPosition(l_speedkmh, 10, 40)

    l_laptime = ac.addLabel(APP_WINDOW, "Lap Time: 0")
    ac.setPosition(l_laptime, 10, 70)

    l_normsplinepos = ac.addLabel(APP_WINDOW, "Normalized Spline Position: 0")
    ac.setPosition(l_normsplinepos, 10, 100)

    l_velocityX = ac.addLabel(APP_WINDOW, "Velocity X: 0")
    ac.setPosition(l_velocityX, 10, 130)

    l_velocityY = ac.addLabel(APP_WINDOW, "Velocity Y: 0")
    ac.setPosition(l_velocityY, 10, 160)

    l_velocityZ = ac.addLabel(APP_WINDOW, "Velocity Z: 0")
    ac.setPosition(l_velocityZ, 10, 190)

    l_worldposX = ac.addLabel(APP_WINDOW, "World Position X: 0")
    ac.setPosition(l_worldposX, 10, 220)

    l_worldposY = ac.addLabel(APP_WINDOW, "World Position Y: 0")
    ac.setPosition(l_worldposY, 10, 250)

    l_worldposZ = ac.addLabel(APP_WINDOW, "World Position Z: 0")
    ac.setPosition(l_worldposZ, 10, 280)

    l_disttraveled = ac.addLabel(APP_WINDOW, "Distance Traveled: 0")
    ac.setPosition(l_disttraveled, 10, 310)

    ac.console("[ACRL] Initialized")
    return APP_NAME


def acUpdate(deltaT):
    global l_speedkmh, l_laptime, l_normsplinepos, l_velocity, l_worldpos, l_disttraveled

    # Update the labels
    speed = ac.getCarState(0, acsys.CS.SpeedKMH)
    ac.setText(l_speedkmh, "Speed (km/h): {}".format(round(speed, DECIMALS)))

    laptime = ac.getCarState(0, acsys.CS.LapTime)
    ac.setText(l_laptime, "Lap Time: {}".format(round(laptime, DECIMALS)))

    splinepos = ac.getCarState(0, acsys.CS.NormalizedSplinePosition)
    ac.setText(l_normsplinepos, "Normalized Spline Position: {}".format(
        round(splinepos, DECIMALS)))

    velocity = ac.getCarState(0, acsys.CS.Velocity)
    ac.setText(l_velocityX, "Velocity X: {}".format(
        round(velocity[0], DECIMALS)))
    ac.setText(l_velocityY, "Velocity Y: {}".format(
        round(velocity[1], DECIMALS)))
    ac.setText(l_velocityZ, "Velocity Z: {}".format(
        round(velocity[2], DECIMALS)))

    worldpos = ac.getCarState(0, acsys.CS.WorldPosition)
    ac.setText(l_worldposX, "World Position X: {}".format(
        round(worldpos[0], DECIMALS)))
    ac.setText(l_worldposY, "World Position Y: {}".format(
        round(worldpos[1], DECIMALS)))
    ac.setText(l_worldposZ, "World Position Z: {}".format(
        round(worldpos[2], DECIMALS)))

    dist_traveled = info.graphics.distanceTraveled
    ac.setText(l_disttraveled, "Distance Traveled: {}".format(
        round(dist_traveled, DECIMALS)))


def acShutdown():
    ac.console("[ACRL] Shutting down...")