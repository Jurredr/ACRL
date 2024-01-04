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
from sim_info import info  # noqa: E402


"""
0 = FL, 1 = FR, 2 = RL, 3 = RR
"""


def get_tyre_wear_value(tyre: int) -> float:
    """
    Retrieve tyre wear of a tyre. 100 is mint condition, 0 is fully worn (puncture)
    :param tyre: int [0,3]
    :return: tyre wear [0,100]
    """
    return info.physics.tyreWear[tyre]


def get_tyre_dirty(tyre: int) -> float:
    """
    Retrieve "dirty level" or a tyre. 0 is clean, 5 is most dirty
    :param tyre: int [0,3]
    :return: dirty level [0,10]
    """
    return info.physics.tyreDirtyLevel[tyre]


def get_tyre_temp(tyre: int, loc: str) -> float:
    """
    Retrieve temperature of a tyre in a location
    :param tyre: int [0,3]
    :param loc: "i" is inner, "m" is middle, "o" is outer, "c" is core temperatures
    :return: temperature of tyre in location
    """
    # Inner
    if loc == "i":
        return info.physics.tyreTempI[tyre]
    # Middle
    elif loc == "m":
        return info.physics.tyreTempM[tyre]
    # Outer
    elif loc == "o":
        return info.physics.tyreTempO[tyre]
    # Core
    elif loc == "c":
        return info.physics.tyreCoreTemperature[tyre]


def get_tyre_pressure(tyre: int) -> float:
    """
    Retrieve tyre pressure of a tyre
    :param tyre: int [0,3]
    :return: tyre pressure
    """
    return info.physics.wheelsPressure[tyre]

# Stays 26.0 for some reason


def get_brake_temp(loc: int = 0) -> float:
    """
    Retrieve temperature of a brake
    :param loc: [0,3]
    :return: brake temperature
    """
    return info.physics.brakeTemp[loc]

# These return a 4D vector

# Slip ratio, between 0 and 1


def get_slip_ratio(car: int = 0):
    return ac.getCarState(car, acsys.CS.SlipRatio)

# Angle of slip, angle between the desired direction and the actual direction of the vehicle [0, 360], degrees


def get_slip_angle(car: int = 0):
    return ac.getCarState(car, acsys.CS.SlipAngle)

# Angle in degrees


def get_camber(car: int = 0):
    return ac.getCarState(car, acsys.CS.CamberDeg)

# Self alligning torque [0, ...]


def get_torque(car: int = 0):
    return ac.getCarState(car, acsys.CS.Mz)

# Load on each tyre [0, ...]


def get_load(car: int = 0):
    return ac.getCarState(car, acsys.CS.Load)

# Vertical suspension travel [0, ...]


def get_suspension_travel(car: int = 0):
    return ac.getCarState(car, acsys.CS.SuspensionTravel)

# Normal vector to tyre's contact point, in x,y,z


def get_tyre_contact_normal(car: int = 0, tyre: int = 0):
    return ac.getCarState(car, acsys.CS.TyreContactNormal, tyre)

# Tyre contact point with the tarmac, in x,y,z


def get_tyre_contact_point(car: int = 0, tyre: int = 0):
    return ac.getCarState(car, acsys.CS.TyreContactPoint, tyre)

# [x,y,z][tyre]


def get_tyre_heading_vector(tyre: int = 0):
    x = info.physics.tyreContactHeading[0][tyre]
    y = info.physics.tyreContactHeading[1][tyre]
    z = info.physics.tyreContactHeading[2][tyre]
    res = (x, y, z)
    return res

# Always returns -1


def get_tyre_right_vector(car: int = 0, tyre: int = 0):
    return ac.getCarState(car, acsys.CS.TyreRightVector, tyre)

# Rad/s


def get_angular_speed(tyre: int = 0):
    return info.physics.wheelAngularSpeed[tyre]
