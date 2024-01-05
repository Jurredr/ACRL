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


def get_has_drs() -> int:
    """
    Retrieves whether car driven by player has DRS
    :return: 0 if no DRS, 1 if there is DRS
    """
    return info.static.hasDRS


def get_has_ers() -> int:
    """
    Retrieves whether car driven by player has ERS
    :return: 0 if no ERS, 1 if there is ERS
    """
    return info.static.hasERS


def get_has_kers() -> int:
    """
    Retrieves whether car driven by player has KERS
    :return: 0 if no KERS, 1 if there is KERS
    """
    return info.static.hasKERS


def abs_level() -> int:
    """
    Retrieves the ABS level active for car driven by player (seems to be buggy)
    :return: value between 0 and 1, the higher, the stronger the ABS
    """
    return info.physics.abs


def get_max_rpm() -> int:
    """
    Retrieves the Maximum RPM of car driven by player
    :return: the maximum RPM
    """
    if info.static.maxRpm:
        return info.static.maxRpm
    else:
        return 1000000


def get_max_fuel() -> int:
    """
    Retrieves the maximum fuel of car driven by player
    :return: the maximum fuel (in KG (or maybe even Liters))
    """
    return info.static.maxFuel
