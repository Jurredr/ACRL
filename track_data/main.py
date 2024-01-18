import csv

world_data = []
velocity_data = []
lap_data = []

# Read car.csv
with open('car.csv', 'r') as car_file:
    car_reader = csv.reader(car_file)
    next(car_reader)  # Skip the header row
    for row in car_reader:
        # Get the world location (second to last value)
        world_location = row[-2]
        velocity = row[9]
        # Convert the string to a tuple of floats
        world_location = tuple(map(float, world_location[1:-1].split(',')))
        velocity = tuple(map(float, velocity[1:-1].split(',')))
        world_data.append(world_location)
        velocity_data.append(velocity)

# Read lap.csv
with open('lap.csv', 'r') as lap_file:
    lap_reader = csv.reader(lap_file)
    next(lap_reader)  # Skip the header row
    for row in lap_reader:
        lap_location = float(row[0])  # Get the lap location (first value)
        lap_data.append(lap_location)

# Create a tick_data list to store all data for every index being the tick number
tick_data = []
for i in range(len(lap_data)):
    tick_data.append([lap_data[i], world_data[i], velocity_data[i]])

# Write tick_data to tick.csv
with open('tick.csv', 'w') as tick_file:
    tick_writer = csv.writer(tick_file)
    tick_writer.writerow(['lap_location', 'world_location'])
    for row in tick_data:
        tick_writer.writerow(row)
