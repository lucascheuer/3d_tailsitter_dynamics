import subprocess
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from traj_gen_circle import generate_circle
from traj_gen_min_snap import generate_minsnap

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


controller_frequency = 1000
traj_speed = 16
waypoints = [
    [0, 0, 0, 0],
    [0, 0, 0, np.pi / 2],
    [0, 8, 0, np.pi / 2],
    [0, 16, 0, 0],
]

start_point_derivatives = [
    [0, 0, 0, 0],  # vel
    [0, 0, 0, 0],  # accl
    [0, 0, 0, 0],  # jerk
    [0, 0, 0, 0],  # snap
]

circle_freq = 1.0 / 10.0
circle_diameter = 10.0
circle_traj = generate_circle(
    traj_speed,
    1.0 / circle_freq,
    circle_freq,
    circle_diameter,
    waypoints[-1][0],
    waypoints[-1][1],
    z_offset=waypoints[-1][2],
    spiral_speed=-0.0,
)

end_point_derivatives = np.zeros((4, 4))
end_point_derivatives[0, 0:3] = circle_traj[0, 4:7]
end_point_derivatives[1, 0:3] = circle_traj[0, 7:10]
end_point_derivatives[2, 0:3] = circle_traj[0, 10:13]
end_point_derivatives[3, 0:3] = circle_traj[0, 15:18]
end_point_derivatives[0, 3] = circle_traj[0, 14]

minsnap_traj_one, wp_one = generate_minsnap(
    waypoints, traj_speed, start_point_derivatives, end_point_derivatives
)


waypoints = [
    [circle_traj[-1, 1], circle_traj[-1, 2], circle_traj[-1, 3], circle_traj[-1, 13]],
    [
        circle_traj[-1, 1],
        circle_traj[-1, 2] + 4,
        circle_traj[-1, 3],
        circle_traj[-1, 13],
    ],
    [
        circle_traj[-1, 1],
        circle_traj[-1, 2] + 8,
        circle_traj[-1, 3],
        circle_traj[-1, 13],
    ],
]

start_point_derivatives = np.zeros((4, 4))
start_point_derivatives[0, 0:3] = circle_traj[-1, 4:7]
start_point_derivatives[1, 0:3] = circle_traj[-1, 7:10]  # / 2
start_point_derivatives[2, 0:3] = circle_traj[-1, 10:13]  # / 6
start_point_derivatives[3, 0:3] = circle_traj[-1, 15:18]  # / 24
start_point_derivatives[0, 3] = circle_traj[-1, 14]

# start_point_derivatives = [
#     [1, 2, 3, 4],  # vel
#     [5, 6, 7, 8],  # accl
#     [9, 1, 2, 3],  # jerk
#     [0, 0, 0, 0],  # snap
# ]
np.set_printoptions(linewidth=np.inf, suppress=True, precision=6)

end_point_derivatives = [
    [0, 0, 0, 0],  # vel
    [0, 0, 0, 0],  # accl
    [0, 0, 0, 0],  # jerk
    [0, 0, 0, 0],  # snap
]

traj_speed = 8
minsnap_traj_two, wp_two = generate_minsnap(
    waypoints, traj_speed, start_point_derivatives, end_point_derivatives
)
minsnap_traj_two_start_time = circle_traj[-1, 0]
minsnap_traj_two[:, 0] += minsnap_traj_two_start_time
wp_two[:, 0] += minsnap_traj_two_start_time
print("circle end")
print(circle_traj[-1, :])
print("des start point derivs")
print(start_point_derivatives)
print("actual start point derivs")

print(minsnap_traj_two[0, :])
traj_header = "times,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,jerk_x,jerk_y,jerk_z,yaw,yaw_dot\n"
wp_header = "t,pos_x,pos_y,pos_z,yaw\n"
# circle_traj = circle_traj[:, :15]
with open(trajectory_file, "w") as f:
    f.write(traj_header)
    np.savetxt(
        f,
        minsnap_traj_one,
        fmt="%.5f",
        delimiter=",",
        comments="",
    )
    np.savetxt(
        f,
        circle_traj,
        fmt="%.5f",
        delimiter=",",
        comments="",
    )
    np.savetxt(
        f,
        minsnap_traj_two,
        fmt="%.5f",
        delimiter=",",
        comments="",
    )

with open(waypoint_file, "w") as f:
    f.write(wp_header)
    np.savetxt(
        f,
        wp_one,
        fmt="%.5f",
        delimiter=",",
        comments="",
    )
    np.savetxt(
        f,
        wp_two,
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
ax = plt.figure().add_subplot(projection="3d")
ax.plot(traj[1, :], -traj[2, :], -traj[3, :])


arrow_length = 0.5
x_d = arrow_length * np.cos(wps[4, :])
y_d = -arrow_length * np.sin(wps[4, :])
z_d = np.zeros(y_d.shape)
ax.quiver(
    wps[1, :],
    -wps[2, :],
    -wps[3, :],
    x_d,
    y_d,
    z_d,
    color="r",
)
ax.grid(True)
ax.set_aspect("equal", adjustable="box")

fig, axs = plt.subplots(5, 1)
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
ax = axs[4]
ax.grid(True)
ax.set_title("sx")
ax.plot(traj[0, :], traj[15, :])

fig, axs = plt.subplots(5, 1)
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
ax = axs[4]
ax.grid(True)
ax.set_title("sx")
ax.plot(traj[0, :], traj[16, :])

fig, axs = plt.subplots(5, 1)
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
ax = axs[4]
ax.grid(True)
ax.set_title("sx")
ax.plot(traj[0, :], traj[17, :])

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
