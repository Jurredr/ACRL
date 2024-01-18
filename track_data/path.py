from cmath import cos, sin
import math
from scipy import optimize
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# In this file, we are going to create a smooth path out of location data.
# The location data is stored in tick.csv, which is created by running main.py.
# In each row in tick.csv, there is a lap_location and a world_location.
# The lap_location is the percentage of the lap that has been completed.
# The world_location is the location of the car in the world.
# The world_location is a tuple of three floats: (x, y, z).
# The x and y values are the coordinates in the world.

# The goal is to create a smooth path out of the world_location values.
# This path is the center line of a racing track.

# Read the tick.csv file
data = pd.read_csv(
    '/Users/jurre/Desktop/University/Year 3/Module 12 (Research Project)/Code/ACRL/track_data/tick.csv')

# Extract lap_location and world_location columns
lap_location = data['lap_location']
world_location = data['world_location']
velocity = data['velocity']

# Create a list of points for the path
world_location = [eval(x) for x in world_location]

velocity = [eval(x) for x in velocity]

# Extract x and z coordinates from world_location
path = np.array([[x, z] for x, y, z in world_location])

# Transpose the path array to get separate x and z arrays
x, y = path.T

okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
xp = np.r_[x[okay], x[-1], x[0]]
yp = np.r_[y[okay], y[-1], y[0]]

# Use splprep to create a spline representation
tck, u = splprep([xp, yp], s=0, per=False)

# Evaluate the spline at 100 points
u_new = np.linspace(0, 1, 1000)
spline_points = splev(u_new, tck)


def get_distance_to_center_line(x, y):
    # Calculate the distance to the center line
    # The distance to the center line is the distance to the closest point on the center line
    # We can calculate this distance by calculating the distance to every point on the center line
    # and then taking the minimum of those distances
    distances = []
    for i in range(len(spline_points[0])):
        # Calculate the distance to the current point
        distance = ((x - spline_points[0][i]) **
                    2 + (y - spline_points[1][i]) ** 2) ** 0.5
        # Add the distance to the list of distances
        distances.append(distance)
    # Return the minimum distance
    return min(distances)


def get_heading_error(x, y, velocity):
    """
    Calculate the difference between the car heading (velocity) and the heading of the center line point closest to the car.
    """

    # Convert velocity values to floats
    velocity = np.array(velocity, dtype=float)

    # Calculate the heading of the car
    heading_car = np.arctan2(velocity[2], velocity[0])

    # Calculate the heading of the center line point closest to the car
    # First, calculate the distance to every point on the center line
    distances = []
    for i in range(len(spline_points[0])):
        # Calculate the distance to the current point
        distance = ((x - spline_points[0][i]) **
                    2 + (y - spline_points[1][i]) ** 2) ** 0.5
        # Add the distance to the list of distances
        distances.append(distance)
    # Find the index of the point with the minimum distance
    index = distances.index(min(distances))
    # Calculate the heading of the closest point
    heading_closest_point = np.arctan2(
        spline_points[1][index] - y, spline_points[0][index] - x)

    # Calculate the difference between the car heading and the heading of the closest point
    heading_error = heading_car - heading_closest_point

    # Make sure the heading error is between -pi and pi
    if heading_error > np.pi:
        heading_error -= 2 * np.pi
    elif heading_error < -np.pi:
        heading_error += 2 * np.pi

    # Return the heading error in radians
    return heading_error


# Example usage:
# Assuming x, y, heading are the current car coordinates and heading
# Replace with actual values
x_car, y_car, heading_vector = xp[0], yp[0], velocity[10]


# Example usage of the functions
distance_to_center = get_distance_to_center_line(x_car, y_car)
heading_error = get_heading_error(x_car, y_car, heading_vector)
# Radians to degrees
print(cos(heading_error), sin(heading_error))
heading_error = math.degrees(heading_error)
print(cos(heading_error), sin(heading_error))
# print(distance_to_center, heading_error)


# Plot the spline
# plt.plot(spline_points[0], spline_points[1])
# plt.plot(x, y, 'ro')
# # Plot the start point (finish line)
# plt.plot(0, 0, 'go')
# plt.show()
