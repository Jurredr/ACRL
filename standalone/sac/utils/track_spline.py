import numpy as np


def _get_closest_spline_point(spline_points, x, y):
    """
    Get the index of the closest point on the spline to the car.
    """
    # First, calculate the distance to every point on the center line
    distances = np.sqrt(
        (x - spline_points[0])**2 + (y - spline_points[1])**2)

    # Find the index of the point with the minimum distance
    index = np.argmin(distances)
    return index


def get_distance_to_center_line(spline_points, x, y):
    """
    Calculate the distance from a coordinate to the center line.
    """
    # Get the closest point on the center line
    index = _get_closest_spline_point(spline_points, x, y)

    # Calculate the distance to the closest point
    distance = np.sqrt((x - spline_points[0][index])**2 +
                       (y - spline_points[1][index])**2)

    # If the distance is smaller than 0.5, we consider it 0
    if distance <= 0.5:
        distance = 0.0
    return distance


def get_heading_error(spline_points, x, y, velocity_vector):
    """
    Calculate the difference between the car heading (velocity) and the heading of the center line point closest to the car.
    """
    # Get the closest point on the center line
    index = _get_closest_spline_point(spline_points, x, y)

    # Get the next spline point
    next_point = index + 5
    if next_point >= len(spline_points[0]):
        next_point = 0

    # Calculate the normalized direction vector from the closest spline point to the next point
    direction_vector = np.array(
        [spline_points[0][next_point] - x, spline_points[1][next_point] - y])
    direction_vector /= np.linalg.norm(direction_vector)

    # Normalize the velocity vector
    velocity_vector /= np.linalg.norm(velocity_vector)

    # Calculate the heading error
    heading_error = np.arccos(
        np.clip(np.dot(direction_vector, velocity_vector), -1.0, 1.0))
    return heading_error
