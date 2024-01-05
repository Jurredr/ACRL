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


def format_time(millis: int) -> str:
    """
    Format time takes an integer representing milliseconds and turns it into a readable string.
    :param millis: the amount of milliseconds
    :return: formatted string [minutes, seconds, milliseconds]
    """
    m = int(millis / 60000)
    s = int((millis % 60000) / 1000)
    ms = millis % 1000

    return "{:02d}:{:02d}.{:03d}".format(m, s, ms)


def get_current_lap_time(car: int = 0, formatted: bool = False):
    """
    Retrieves the current lap time of the car selected
    :param car: the car selected (user is 0)
    :param formatted: true if format should be in readable str
    :return: current lap time in milliseconds (int) or string format
    """
    if formatted:
        time = ac.getCarState(car, acsys.CS.LapTime)
        if time > 0:
            return format_time(time)
        else:
            return "--:--"
    else:
        return ac.getCarState(car, acsys.CS.LapTime)


def get_last_lap_time(car: int = 0, formatted: bool = False):
    """
    Retrieves the last lap time of the car selected
    :param car: the car selected (user is 0)
    :param formatted: true if format should be in readable str
    :return: last lap time in milliseconds (int) or string format
    """
    if formatted:
        time = ac.getCarState(car, acsys.CS.LastLap)
        if time > 0:
            return format_time(time)
        else:
            return "--:--"
    else:
        return ac.getCarState(car, acsys.CS.LastLap)


def get_best_lap_time(car: int = 0, formatted: bool = False):
    """
    Retrieve the best lap time recorded, does not save if invalidated lap
    :param car: the car selected (user is 0)
    :param formatted: true if format should be in readable str
    :return: best lap time in string format or formatted string
    """
    if formatted:
        time = ac.getCarState(car, acsys.CS.BestLap)
        if time > 0:
            return format_time(time)
        else:
            return "--:--"
    else:
        return ac.getCarState(car, acsys.CS.BestLap)


def get_splits(car: int = 0, formatted: bool = False):
    """
    Retrieve the split times of the completed lap
    :param car: the car selected (user is 0)
    :param formatted: true if format should be in readable str
    :return: list containing the splits in milliseconds (int) or string format
    """
    if formatted:
        times = ac.getLastSplits(car)
        formattedtimes = []

        if len(times) != 0:
            for t in times:
                formattedtimes.append(format_time(t))
            return formattedtimes
        else:
            return "--:--"
    else:
        return ac.getLastSplits(car)


def get_split() -> str:
    """
    Retrieve the last sector split, but will return nothing if the last sector is the completion of a lap
    :return: split in string format
    """
    return info.graphics.split


def get_invalid(car: int = 0) -> bool:
    """
    Retrieve if the current lap is invalid
    :param car: the car selected (user is 0)
    :return: Invalid lap in boolean form
    """
    import ac_api.car_info as ci

    return ac.getCarState(car, acsys.CS.LapInvalidated) or ci.get_tyres_off_track() > 2


def get_lap_count(car: int = 0) -> int:
    """
    Retrieve the current number of laps
    :param car: the car selected (user is 0)
    :return: The current number of laps (added by 1 default)
    """
    return ac.getCarState(car, acsys.CS.LapCount) + 1


def get_laps() -> str:
    """
    Returns the total number of laps in a race (only in a race)
    :return: total number of race laps
    """
    if info.graphics.numberOfLaps > 0:
        return info.graphics.numberOfLaps
    else:
        return "-"


def get_lap_delta(car: int = 0) -> float:
    """
    Retrieves the delta to the fastest lap
    :param car: the car selected (user is 0)
    :return: delta to the fastest lap in seconds (float)
    """
    return ac.getCarState(car, acsys.CS.PerformanceMeter)

# Either 0, 1 or 2


def get_current_sector():
    return info.graphics.currentSectorIndex
