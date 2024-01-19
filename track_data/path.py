from cmath import cos, sin
from math import degrees
from matplotlib.animation import FuncAnimation
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

    distances = np.sqrt(
        (x - spline_points[0])**2 + (y - spline_points[1])**2)

    # Find the index of the point with the minimum distance
    return np.min(distances)


def update(frame):
    # Clear the previous plot
    plt.cla()

    # Plot the spline
    plt.plot(spline_points[0], spline_points[1])
    plt.plot(x, y, 'ro')

    # Skip every uneven frame
    time_step = frame * 10
    x_car, y_car = xp[time_step], yp[time_step]

    # Plot the start point (finish line)
    plt.plot(x_car, y_car, 'go')

    # First, calculate the distance to every point on the center line
    distances = np.sqrt(
        (x_car - spline_points[0])**2 + (y_car - spline_points[1])**2)

    # Find the index of the point with the minimum distance
    index = np.argmin(distances)

    # Plot the point on the center line closest to the car
    plt.plot(spline_points[0][index], spline_points[1]
             [index], 'bo', markersize=10, zorder=-1)

    # Get the next spline point and draw a line between the car and that point
    next_point = index + 5
    if next_point >= len(spline_points[0]):
        next_point = 0

    # Calculate the normalized direction vector from the car to the next point
    direction_vector = np.array(
        [spline_points[0][next_point] - x_car, spline_points[1][next_point] - y_car])
    direction_vector /= np.linalg.norm(direction_vector)

    # Plot the direction vector
    plt.arrow(x_car, y_car, direction_vector[0] * 100,
              direction_vector[1] * 100, color='y', zorder=15, width=2)

    car_velocity = velocity[time_step]
    # Change the car_velocity to a 2d vector
    car_velocity = np.array([car_velocity[0], car_velocity[2]])
    car_velocity /= np.linalg.norm(car_velocity)

    # Plot the velocity vector
    plt.arrow(x_car, y_car, car_velocity[0] * 100,
              car_velocity[1] * 100, color='purple', zorder=15, width=2)

    # Print the heading error (difference between the direction_vector and the car_velocity)
    heading_error = np.arccos(
        np.clip(np.dot(direction_vector, car_velocity), -1.0, 1.0))

    straightness = cos(heading_error).real - sin(heading_error).real


# Create the animation
animation = FuncAnimation(plt.gcf(), update, frames=len(xp), interval=1)

# Show the animation
plt.show()
