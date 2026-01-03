import numpy as np
import csv

# Define the data
header_text = "times,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,jerk_x,jerk_y,jerk_z,yaw,yaw_dot"
circle_amplitude = 16.0
circle_frequency = 1.0 / 16.0
spiral_speed = 0.0
t_start = 0.0
t_end = 16.0
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

# pos_y = np.zeros(times.shape)
# vel_y = np.zeros(times.shape)
# acc_y = np.zeros(times.shape)
# jerk_y = np.zeros(times.shape)

pos_z = times * spiral_speed
vel_z = np.ones(times.shape) * spiral_speed
acc_z = np.zeros(times.shape)
jerk_z = np.zeros(times.shape)
# yaw = np.zeros(times.shape)
# yaw_dot = np.zeros(times.shape)
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
print("pos", data[0, 1:4])
print("vel", data[0, 4:7])
print("yaw, yaw_rate", data[0, -2:])
np.savetxt(
    "param_files/trajectory.csv",
    data,
    fmt="%.9e",
    delimiter=",",
    header=header_text,
    comments="",
)
