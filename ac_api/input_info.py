import os
import sys
import platform

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
import acsys  # noqa: E402


def get_gas_input(car: int = 0) -> float:
    """
    Retrieve the gas input given to a car
    :param car: the car selected (user is 0)
    :return: gas input between 0 and 1
    """
    return ac.getCarState(car, acsys.CS.Gas)


def get_brake_input(car: int = 0) -> float:
    """
    Retrieve the brake input given to a car
    :param car: the car selected (user is 0)
    :return: brake input between 0 and 1
    """
    return ac.getCarState(car, acsys.CS.Brake)


def get_clutch(car: int = 0) -> float:
    """
    Retrieve the clutch status in the game of a car
    :param car: the car selected (user is 0)
    :return: deployment of the clutch (1 is fully deployed, 0 is not deployed).
    """
    return ac.getCarState(car, acsys.CS.Clutch)


def get_steer_input(car: int = 0) -> float:
    """
    Retrieve the steering input given to a car
    :param car: the car selected (user is 0)
    :return: steering input to the car, depends on the settings in AC, in degrees
    """
    return ac.getCarState(car, acsys.CS.Steer)

# Test on not laptop [0, ...]


def get_last_ff(car: int = 0) -> float:
    return ac.getCarState(car, acsys.CS.LastFF)
