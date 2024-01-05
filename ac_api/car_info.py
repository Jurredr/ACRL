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
    Formats the time from timestamp to readable time
    :param millis: timestamp in milliseconds
    :return: formatted string in format mm:ss:ms
    """
    m = int(millis / 60000)
    s = int((millis % 60000) / 1000)
    ms = millis % 1000

    return "{:02d}:{:02d}.{:03d}".format(m, s, ms)


def get_speed(car: int = 0, unit: str = "kmh") -> float:
    """
    Retrieve the current speed of a car
    :param car: the car selected (user is 0)
    :param unit: either kmh or mph or ms on how to show speed
    :return: current speed [0, ...]
    """
    if unit == "kmh":
        return ac.getCarState(car, acsys.CS.SpeedKMH)
    elif unit == "mph":
        return ac.getCarState(car, acsys.CS.SpeedMPH)
    elif unit == "ms":
        return ac.getCarState(car, acsys.CS.SpeedMS)


def get_delta_to_car_ahead(formatted: bool = False):
    """
    Retrieve time delta to the car ahead
    :param formatted: true if format should be in readable str
    :return: delta to car ahead in calculated time distance (float) or string format
    """
    import ac_api.session_info as si
    import ac_api.lap_info as li
    time = 0
    dist = 0
    track_len = si.get_track_length()
    lap = li.get_lap_count(0)
    pos = get_location(0)

    for car in range(si.get_cars_count()):
        if get_position(car) == get_position(0) - 1:
            lap_next = li.get_lap_count(car)
            pos_next = get_location(car)

            dist = max(0, (pos_next * track_len + lap_next *
                       track_len) - (pos * track_len + lap * track_len))
            time = max(0.0, dist / max(10.0, get_speed(0, "ms")))
            break

    if not formatted:
        return time
    else:
        if dist > track_len:
            laps = dist / track_len
            if laps > 1:
                return "+{:3.1f}".format(laps) + " Laps"
            else:
                return "+{:3.1f}".format(laps) + " Lap"
        elif time > 60:
            return "+" + format_time(int(time * 1000))
        else:
            return "+{:3.3f}".format(time)


def get_delta_to_car_behind(formatted: bool = False):
    """
    Retrieve time delta to the car behind
    :param formatted: true if format should be in readable str
    :return: delta to car behind in calculated time distance (float) or string format
    """
    import ac_api.session_info as si
    import ac_api.lap_info as li
    time = 0
    dist = 0
    track_len = si.get_track_length()
    lap = li.get_lap_count(0)
    pos = get_location(0)
    for car in range(si.get_cars_count()):
        if get_position(car) == get_position(0) + 1:
            lap_next = li.get_lap_count(car)
            pos_next = get_location(car)

            dist = max(0, (pos * track_len + lap * track_len) -
                       (pos_next * track_len + lap_next * track_len))
            time = max(0.0, dist / max(10.0, get_speed(car, "ms")))
            break

    if not formatted:
        return time
    else:
        if dist > track_len:
            laps = dist / track_len
            if laps > 1:
                return "-{:3.1f}".format(laps) + " Laps"
            else:
                return "-{:3.1f}".format(laps) + " Lap"
        elif time > 60:
            return "-" + format_time(int(time * 1000))
        else:
            return "-{:3.3f}".format(time)


def get_location(car: int = 0) -> float:
    """
    Retrieve current location of a car
    :param car: the car selected (user is 0)
    :return: position on track relative with the lap between 0 and 1
    """
    return ac.getCarState(car, acsys.CS.NormalizedSplinePosition)


def get_world_location(car: int = 0):
    """
    Retrieve absolute location of a car
    :param car: the car selected (user is 0)
    :return: absolute location [x,y,z] ((0,x,0) is the middle)
    """
    x = ac.getCarState(car, acsys.CS.WorldPosition)[0]
    y = ac.getCarState(car, acsys.CS.WorldPosition)[1]
    z = ac.getCarState(car, acsys.CS.WorldPosition)[2]
    res = (x, y, z)
    return res


def get_position(car: int = 0) -> int:
    """
    Retrieve current driving position of a car
    :param car: the car selected (user is 0)
    :return: position of car (0 is the lead car)
    """
    return ac.getCarRealTimeLeaderboardPosition(car) + 1


def get_drs_available():
    return info.physics.drsAvailable

# 0 if disabled, 1 if enabled


def get_drs_enabled() -> bool:
    """
    Check whether DRS of the car of the player is enabled
    :return: DRS enabled
    """
    return info.physics.drsEnabled

# Formatted: 0=R, 1=N, 2=1, 3=2, 4=3, 5=4, 6=5, 7=6, 8=7, etc.


def get_gear(car: int = 0, formatted: bool = True):
    """
    Retrieve current gear of a car. if Formatted, it returns string, if not, it returns int. 0=R, 1=N, 2=1, 3=2, etc.
    :param car: the car selected (user is 0)
    :param formatted: boolean to format result or not.
    :return: current gear of car as integer or string format
    """
    gear = ac.getCarState(car, acsys.CS.Gear)
    if formatted:
        if gear == 0:
            return "R"
        elif gear == 1:
            return "N"
        else:
            return str(gear - 1)
    else:
        return gear


def get_rpm(car: int = 0) -> float:
    """
    Retrieve rpm of a car
    :param car: the car selected (user is 0)
    :return: rpm of a car [0, ...]
    """
    return ac.getCarState(car, acsys.CS.RPM)


def get_fuel() -> float:
    """
    Retrieve amount of fuel in player's car in kg
    :return: amount of fuel [0, ...]
    """
    return info.physics.fuel

# Returns the amount of tyres off-track


def get_tyres_off_track() -> int:
    """
    Retrieve amount of tyres of player's car off-track
    :return: amount of tyres off-track [0,4]
    """
    return info.physics.numberOfTyresOut


def get_car_in_pit_lane() -> bool:
    """
    Retrieve whether player's car is in the pitlane
    :return: car in pit lane
    """
    return info.graphics.isInPitLane


# Damage numbers go up to a high number. A slight tap results in a damage value of about 10
def get_location_damage(loc: str = "front") -> float:
    """
    Retrieve car damage per side
    :param loc: front, rear, left or right
    :return: damage [0, ...]
    """
    if loc == "front":
        return info.physics.carDamage[0]
    elif loc == "rear":
        return info.physics.carDamage[1]
    elif loc == "left":
        return info.physics.carDamage[2]
    elif loc == "right":
        return info.physics.carDamage[3]
    else:
        # Centre
        return info.physics.carDamage[4]


def get_total_damage():
    front = info.physics.carDamage[0]
    rear = info.physics.carDamage[1]
    left = info.physics.carDamage[2]
    right = info.physics.carDamage[3]
    centre = info.physics.carDamage[4]
    res = (front, rear, left, right, centre)
    return res

# Height of the center of gravity of the car from the ground [0, ...]


def get_cg_height(car: int = 0) -> float:
    return ac.getCarState(car, acsys.CS.CGHeight)

# Speed Delivered to the wheels [0, ...]. Difference between actual speed might cause engine braking?


def get_drive_train_speed(car: int = 0):
    return ac.getCarState(car, acsys.CS.DriveTrainSpeed)

# Returns velocity in coordinates x,y,z


def get_velocity():
    x = info.physics.velocity[0]
    y = info.physics.velocity[1]
    z = info.physics.velocity[2]
    res = (x, y, z)
    return res


def get_acceleration():
    x = info.physics.accG[0]
    y = info.physics.accG[1]
    z = info.physics.accG[2]
    res = (x, y, z)
    return res

# Checks whether tc is needed


def get_tc_in_action():
    return info.physics.tc

# 0 for false, 1 for true


def get_abs_in_action():
    return info.physics.abs

# Front brake bias between 0(%) and 1(00%)


def get_brake_bias():
    return info.physics.brakeBias

# Different engine brake mappings


def get_engine_brake():
    return info.physics.engineBrake
