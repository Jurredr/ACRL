import sys
import os
import platform
import threading

# The name of the app
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

# Import the Assetto Corsa libraries
import ac  # noqa: E402
import acsys  # noqa: E402
from sim_info import info  # noqa: E402
from IS_ACUtil import *  # noqa: E402

# Value labels
l_speedkmh = 0
l_laptime = 0
l_laptimeInvalid = 0
l_lapFinished = 0
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
    ac.setSize(APP_WINDOW, 320, 390)
    ac.setTitle(APP_WINDOW, APP_NAME +
                ": Reinforcement Learning")

    # Background fully black
    ac.setBackgroundOpacity(APP_WINDOW, 1)

    # Create the labels
    global l_speedkmh, l_laptime, l_laptimeInvalid, l_lapFinished, l_velocityX, l_velocityY, l_velocityZ, l_worldposX, l_worldposY, l_worldposZ, l_disttraveled

    l_speedkmh = ac.addLabel(APP_WINDOW, "Speed (km/h): 0")
    ac.setPosition(l_speedkmh, 10, 40)

    l_laptime = ac.addLabel(APP_WINDOW, "Lap Time: 0")
    ac.setPosition(l_laptime, 10, 70)

    l_laptimeInvalid = ac.addLabel(APP_WINDOW, "Lap Time Invalid: False")
    ac.setPosition(l_laptimeInvalid, 10, 100)

    l_lapFinished = ac.addLabel(APP_WINDOW, "Lap Finished: False")
    ac.setPosition(l_lapFinished, 10, 130)

    l_velocityX = ac.addLabel(APP_WINDOW, "Velocity X: 0")
    ac.setPosition(l_velocityX, 10, 160)

    l_velocityY = ac.addLabel(APP_WINDOW, "Velocity Y: 0")
    ac.setPosition(l_velocityY, 10, 190)

    l_velocityZ = ac.addLabel(APP_WINDOW, "Velocity Z: 0")
    ac.setPosition(l_velocityZ, 10, 220)

    l_worldposX = ac.addLabel(APP_WINDOW, "World Position X: 0")
    ac.setPosition(l_worldposX, 10, 250)

    l_worldposY = ac.addLabel(APP_WINDOW, "World Position Y: 0")
    ac.setPosition(l_worldposY, 10, 280)

    l_worldposZ = ac.addLabel(APP_WINDOW, "World Position Z: 0")
    ac.setPosition(l_worldposZ, 10, 310)

    l_disttraveled = ac.addLabel(APP_WINDOW, "Distance Traveled: 0")
    ac.setPosition(l_disttraveled, 10, 340)

    ac.console("[ACRL] Initialized")
    return APP_NAME


def acUpdate(deltaT):
    global l_speedkmh, l_laptime, l_laptimeInvalid, l_lapFinished, l_velocityX, l_velocityY, l_velocityZ, l_worldposX, l_worldposY, l_worldposZ, l_disttraveled

    # Update the labels
    # Speed (km/h)
    speed = ac.getCarState(0, acsys.CS.SpeedKMH)
    ac.setText(l_speedkmh, "Speed (km/h): {}".format(round(speed, DECIMALS)))

    # Lap time
    laptime = ac.getCarState(0, acsys.CS.LapTime)
    ac.setText(l_laptime, "Lap Time: {}".format(round(laptime, DECIMALS)))

    # Lap Time Invalid
    laptime_invalid = ac.getCarState(0, info.physics.numberOfTyresOut)
    ac.setText(l_laptimeInvalid, "Lap Time Invalid: {}".format(laptime_invalid))

    # Lap Finished
    lap_finished = ac.getCarState(0, acsys.CS.LapCount)
    ac.setText(l_lapFinished, "Lap Finished: {}".format(str(lap_finished > 0)))

    # Velocity
    velocity = ac.getCarState(0, acsys.CS.Velocity)
    velocity_x = velocity[0]
    velocity_y = velocity[1]
    velocity_z = velocity[2]
    ac.setText(l_velocityX, "Velocity X: {}".format(
        round(velocity_x, DECIMALS)))
    ac.setText(l_velocityY, "Velocity Y: {}".format(
        round(velocity_y, DECIMALS)))
    ac.setText(l_velocityZ, "Velocity Z: {}".format(
        round(velocity_z, DECIMALS)))

    # World position
    worldpos = ac.getCarState(0, acsys.CS.WorldPosition)
    worldpos_x = worldpos[0]
    worldpos_y = worldpos[1]
    worldpos_z = worldpos[2]
    ac.setText(l_worldposX, "World Position X: {}".format(
        round(worldpos_x, DECIMALS)))
    ac.setText(l_worldposY, "World Position Y: {}".format(
        round(worldpos_y, DECIMALS)))
    ac.setText(l_worldposZ, "World Position Z: {}".format(
        round(worldpos_z, DECIMALS)))

    # Distance traveled
    dist_traveled = info.graphics.distanceTraveled
    ac.setText(l_disttraveled, "Distance Traveled: {}".format(
        round(dist_traveled, DECIMALS)))

    t_kh = threading.Thread(target=keyhook)
    t_kh.start()


def keyhook():
    while True:
        if getKeyState(17):
            if getKeyState(48):
                ac.console("[ACRL] Ctrl + 0 pressed")


def acShutdown():
    ac.console("[ACRL] Shutting down...")
