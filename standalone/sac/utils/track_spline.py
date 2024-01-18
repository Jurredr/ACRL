import numpy as np


def get_distance_to_center_line(spline_points, x, y):
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


def get_heading_error(spline_points, x, y, velocity):
    """
    Calculate the difference between the car heading (velocity) and the heading of the center line point closest to the car.
    """

    # Convert velocity values to floats
    velocity = np.array(velocity, dtype=float)

    # Calculate the heading of the car
    heading_car = np.arctan2(velocity[1], velocity[0])

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
