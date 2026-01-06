import subprocess
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from traj_gen_circle import generate_circle

traj_gen_path = (
    Path(__file__).resolve().parent.parent
    / "cpp"
    / "trajectory_generation"
    / "build"
    / "min_snap_generator"
)
out_folder = Path(__file__).resolve().parent.parent / "trajectory_files"
waypoint_folder = Path(__file__).resolve().parent.parent / "waypoint_files"

# output files
trajectory_file = out_folder / "trajectory_test.csv"
waypoint_file = waypoint_folder / "waypoint_test.csv"
fig, ax = plt.subplots(1, 1)
fig.suptitle("XY")

controller_frequency = 1000
traj_speed = 16
speed_weight = 0
descent_rate = 0.0001

step_limit = 100
waypoints = [
    [0, 0, 0, 0],
    [0, 0, 0, np.pi / 2],
    [0, 16, 0, np.pi / 2],
    [0, 32, 0, 0],
    # [0, 2, 0, np.pi * 1.2],
]

start_point_derivatives = [
    [0, 0, 0, 0],  # vel
    [0, 0, 0, 0],  # accl
    [0, 0, 0, 0],  # jerk
    [0, 0, 0, 0],  # snap
]

use_circle = True
circle_freq = 1.0 / 10.0
circle_diameter = 8.0
if use_circle:
    circle_data = generate_circle(
        traj_speed,
        1.0 / circle_freq,
        circle_freq,
        circle_diameter,
        waypoints[-1][0],
        waypoints[-1][1],
        z_offset=waypoints[-1][2],
        spiral_speed=-0.1,
    )

    end_point_derivatives = np.zeros((4, 4))
    end_point_derivatives[0, 0:3] = circle_data[-1, 4:7]
    end_point_derivatives[1, 0:3] = circle_data[-1, 7:10]
    end_point_derivatives[2, 0:3] = circle_data[-1, 10:13]
    end_point_derivatives[3, 0:3] = circle_data[-1, 15:18]
    end_point_derivatives[0, 3] = circle_data[-1, 14]
    circle_data = circle_data[:, :15]
else:
    end_point_derivatives = [
        [0, 0, 0, 0],  # vel
        [0, 0, 0, 0],  # accl
        [0, 0, 0, 0],  # jerk
        [0, 0, 0, 0],  # snap
    ]
# angle_step = np.pi / 8
# diameter = 16
# lateral_offset = 0.1
# angles = np.arange(0, np.pi * 2 + angle_step, angle_step)
# xs = 0.5 * diameter * np.cos(angles - np.pi / 2) + lateral_offset
# ys = 0.5 * diameter * np.sin(angles - np.pi / 2) + diameter * 0.5
# waypoints = [[0, 0, 0, 0]]
# for x, y, angle in zip(xs, ys, angles):
#     waypoints.append([x, y, 0, angle])
# waypoints.append([lateral_offset + lateral_offset, 0, 0, np.pi * 4.0])
# waypoints = [
#     [0, 0, 0, 0],
#     [8, 0, 0, 0],
#     [12, 4, 0, np.pi * 0.5],
#     [8, 8, 0, np.pi],
#     [4, 4, 0, np.pi * 1.5],
#     [8, 0, 0, np.pi * 2],
#     [12, 4, 0, np.pi * 2.5],
#     [8, 8, 0, np.pi * 3],
#     [4, 4, 0, np.pi * 3.5],
#     [8, 0, 0, np.pi * 4.0],
#     [16, 0, 0, np.pi * 4.0],
# ]

input_list = [
    traj_gen_path,
    trajectory_file,
    waypoint_file,
    str(controller_frequency),
    str(0),
    str(traj_speed),
    str(speed_weight),
    str(descent_rate),
    str(step_limit),
    str(len(waypoints)),
]
for waypoint in waypoints:
    for item in waypoint:
        input_list.append(str(item))
for derivative in start_point_derivatives:
    for item in derivative:
        input_list.append(str(item))
for derivative in end_point_derivatives:
    for item in derivative:
        input_list.append(str(item))

for inp in input_list:
    print(inp, end=" ")
print()
subprocess.run(input_list)

if use_circle:
    with open(trajectory_file, "a") as f:
        np.savetxt(
            f,
            circle_data,
            fmt="%.5f",
            delimiter=",",
            comments="",
        )

with open(trajectory_file, mode="r", newline="") as file:
    csv_reader = csv.reader(file)
    traj = []
    next(csv_reader)
    for row in csv_reader:
        float_row = [float(item) for item in row]
        traj.append(float_row)

    traj = np.array(traj).T
with open(waypoint_file, mode="r", newline="") as file:
    csv_reader = csv.reader(file)
    wps = []
    next(csv_reader)
    for row in csv_reader:
        float_row = [float(item) for item in row]
        wps.append(float_row)

wps = np.array(wps).T
print("final time:", wps[0, -1])
# fig, ax = plt.subplots(1, 1)
# fig.suptitle("XY")
ax.plot(traj[1, :], traj[2, :])


arrow_length = 0.5
x_d = arrow_length * np.cos(wps[4, :])
y_d = arrow_length * np.sin(wps[4, :])
ax.quiver(
    wps[1, :], wps[2, :], x_d, y_d, scale_units="xy", angles="xy", color="r", scale=1
)
ax.grid(True)
ax.set_aspect("equal", adjustable="box")

fig, axs = plt.subplots(4, 1)
fig.suptitle("X")
ax = axs[0]
ax.grid(True)
ax.set_title("x")
ax.plot(traj[0, :], traj[1, :])
ax.scatter(wps[0, :], wps[1, :], color="r")
ax = axs[1]
ax.grid(True)
ax.set_title("vx")
ax.plot(traj[0, :], traj[4, :])
ax = axs[2]
ax.grid(True)
ax.set_title("ax")
ax.plot(traj[0, :], traj[7, :])
ax = axs[3]
ax.grid(True)
ax.set_title("jx")
ax.plot(traj[0, :], traj[10, :])

fig, axs = plt.subplots(4, 1)
fig.suptitle("Y")
ax = axs[0]
ax.grid(True)
ax.set_title("y")
ax.plot(traj[0, :], traj[2, :])
ax.scatter(wps[0, :], wps[2, :], color="r")
ax = axs[1]
ax.grid(True)
ax.set_title("vy")
ax.plot(traj[0, :], traj[5, :])
ax = axs[2]
ax.grid(True)
ax.set_title("ay")
ax.plot(traj[0, :], traj[8, :])
ax = axs[3]
ax.grid(True)
ax.set_title("jy")
ax.plot(traj[0, :], traj[11, :])

fig, axs = plt.subplots(4, 1)
fig.suptitle("Z")
ax = axs[0]
ax.grid(True)
ax.set_title("z")
ax.plot(traj[0, :], traj[3, :])
ax.scatter(wps[0, :], wps[3, :], color="r")
ax = axs[1]
ax.grid(True)
ax.set_title("vz")
ax.plot(traj[0, :], traj[6, :])
ax = axs[2]
ax.grid(True)
ax.set_title("az")
ax.plot(traj[0, :], traj[9, :])
ax = axs[3]
ax.grid(True)
ax.set_title("jz")
ax.plot(traj[0, :], traj[12, :])

fig, axs = plt.subplots(2, 1)
fig.suptitle("Yaw")
ax = axs[0]
ax.grid(True)
ax.set_title("yaw")
ax.plot(traj[0, :], traj[13, :])
ax.scatter(wps[0, :], wps[4, :], color="r")
ax = axs[1]
ax.grid(True)
ax.set_title("yaw rate")
ax.plot(traj[0, :], traj[14, :])


plt.show()
