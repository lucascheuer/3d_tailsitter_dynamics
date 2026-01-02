import numpy as np
import csv

# Define the data
header_text = "times,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,jerk_x,jerk_y,jerk_z,yaw,yaw_dot"


# Open the file in write mode ('w') with `newline=''`
# Using a 'with' statement ensures the file is closed automatically
circle_amplitude = 2.0
circle_frequency = 1.0 / 8.0
spiral_speed = 0
t_start = 0.0
t_end = 20.0
t_step = 0.001
times = np.arange(t_start, t_end, t_step)
pos_x = circle_amplitude * np.sin(np.pi * 2.0 * circle_frequency * times)
vel_x = (
    circle_amplitude
    * np.pi
    * 2.0
    * circle_frequency
    * np.cos(np.pi * 2.0 * circle_frequency * times)
)
acc_x = (
    -circle_amplitude
    * np.pi**2
    * 2.0**2
    * circle_frequency**2
    * np.sin(np.pi * 2.0 * circle_frequency * times)
)
jerk_x = (
    -circle_amplitude
    * np.pi**3
    * 2.0**3
    * circle_frequency**3
    * np.cos(np.pi * 2.0 * circle_frequency * times)
)

pos_y = circle_amplitude * np.cos(np.pi * 2.0 * circle_frequency * times)
vel_y = (
    -circle_amplitude
    * np.pi
    * 2.0
    * circle_frequency
    * np.sin(np.pi * 2.0 * circle_frequency * times)
)
acc_y = (
    -circle_amplitude
    * np.pi**2
    * 2.0**2
    * circle_frequency**2
    * np.cos(np.pi * 2.0 * circle_frequency * times)
)
jerk_y = (
    circle_amplitude
    * np.pi**3
    * 2.0**3
    * circle_frequency**3
    * np.sin(np.pi * 2.0 * circle_frequency * times)
)
pos_z = times * spiral_speed
vel_z = np.ones(times.shape) * spiral_speed
acc_z = np.zeros(times.shape)
jerk_z = np.zeros(times.shape)

yaw = -times * np.pi * 2.0 * circle_frequency  # + np.pi / 2
yaw_dot = np.ones(times.shape) * -np.pi * 2.0 * circle_frequency
data = np.column_stack(
    (
        times,
        pos_x,
        pos_y,
        pos_z,
        vel_x,
        vel_y,
        vel_z,
        acc_x,
        acc_y,
        acc_z,
        jerk_x,
        jerk_y,
        jerk_z,
        yaw,
        yaw_dot,
    )
)
print(data[0, :])
np.savetxt(
    "trajectory.csv", data, fmt="%.9e", delimiter=",", header=header_text, comments=""
)
# with open("trajectory.csv", mode="w", newline="") as file:
#     writer = csv.writer(file)

#     # Write the header
#     writer.writerow(header)

#     # Write the data row
#     # writer.writerow(data)
