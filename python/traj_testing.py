import subprocess
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from trajectory_generators.traj_gen_circle import generate_circle
from trajectory_generators.traj_gen_min_snap import generate_minsnap


out_folder = Path(__file__).resolve().parent.parent / "trajectory_files"
waypoint_folder = Path(__file__).resolve().parent.parent / "waypoint_files"

# output files
trajectory_file = out_folder / "trajectory_test.csv"
waypoint_file = waypoint_folder / "waypoint_test.csv"


controller_frequency = 1000
traj_speed = 16
waypoints = [
    # [0, 0, 0, 0],
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
end_point_derivatives[3, 0:3] = circle_traj[0, 13:16]
end_point_derivatives[0, 3] = circle_traj[0, 17]

minsnap_traj_one, wp_one = generate_minsnap(
    waypoints, traj_speed, start_point_derivatives, end_point_derivatives
)


waypoints = [
    [circle_traj[-1, 1], circle_traj[-1, 2], circle_traj[-1, 3], circle_traj[-1, 16]],
    [
        circle_traj[-1, 1],
        circle_traj[-1, 2] + 4,
        circle_traj[-1, 3],
        circle_traj[-1, 16] - np.pi / 2,
    ],
    [
        circle_traj[-1, 1],
        circle_traj[-1, 2] + 8,
        circle_traj[-1, 3],
        circle_traj[-1, 16] - np.pi,
    ],
]

start_point_derivatives = np.zeros((4, 4))
start_point_derivatives[0, 0:3] = circle_traj[-1, 4:7]
start_point_derivatives[1, 0:3] = circle_traj[-1, 7:10]
start_point_derivatives[2, 0:3] = circle_traj[-1, 10:13]
start_point_derivatives[3, 0:3] = circle_traj[-1, 13:16]
start_point_derivatives[0, 3] = circle_traj[-1, 17]
np.set_printoptions(linewidth=np.inf, suppress=True, precision=100)
end_point_derivatives = [
    [0, 0, 0, 0],  # vel
    [0, 0, 0, 0],  # accl
    [0, 0, 0, 0],  # jerk
    [0, 0, 0, 0],  # snap
]

traj_speed = 10
minsnap_traj_two, wp_two = generate_minsnap(
    waypoints, traj_speed, start_point_derivatives, end_point_derivatives
)
minsnap_traj_two_start_time = circle_traj[-1, 0]
minsnap_traj_two[:, 0] += minsnap_traj_two_start_time
wp_two[:, 0] += minsnap_traj_two_start_time

traj_header = "times,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,jerk_x,jerk_y,jerk_z,snap_x,snap_y,snap_z,yaw,yaw_dot\n"
wp_header = "t,pos_x,pos_y,pos_z,yaw\n"

final_traj = np.concatenate((minsnap_traj_one, circle_traj, minsnap_traj_two), axis=0)
yaw = np.full(final_traj[:, 5].shape, np.nan)

xy_speed = np.sqrt(final_traj[:, 5] ** 2 + final_traj[:, 4] ** 2)
xy_acc = np.sqrt(final_traj[:, 8] ** 2 + final_traj[:, 7] ** 2)
xy_jerk = np.sqrt(final_traj[:, 11] ** 2 + final_traj[:, 10] ** 2)
xy_snap = np.sqrt(final_traj[:, 14] ** 2 + final_traj[:, 13] ** 2)
mask = xy_speed > 0.1
yaw[mask] = np.atan2(final_traj[mask, 5], final_traj[mask, 4])
yaw_d = np.zeros(yaw.shape)
dt = final_traj[1, 0] - final_traj[0, 0]
yaw_d[:-1] = (yaw[1:] - yaw[:-1]) / dt
mask = abs(yaw[1:] - yaw[:-1]) > np.pi

nan_mask = np.where(np.isnan(yaw))[0]

end_change_mask = np.where(nan_mask[1:] != nan_mask[:-1] + 1)[0]

start_change_mask = end_change_mask + 1
nan_mask_start = nan_mask[start_change_mask]
nan_mask_end = nan_mask[end_change_mask]
print(nan_mask_start)
print(nan_mask_end)
if nan_mask_end[0] < nan_mask_start[0]:
    yaw[0 : nan_mask_end[0]] = np.linspace(0, yaw[nan_mask_end[0] + 1], nan_mask_end[0])
    nan_mask_end = np.delete(nan_mask_end, 0)

if nan_mask_start[-1] > nan_mask_end[-1]:
    print(np.linspace(yaw[nan_mask_start[-1] - 1], 0, nan_mask_start[-1]))
    yaw[nan_mask_start[-1] : -1] = yaw[nan_mask_start[-1] - 1]
    # np.linspace(yaw[nan_mask_start[-1] - 1], 0, yaw.shape[0] - nan_mask_start[-1] - 1)
    # np.arange()
    nan_mask_start = np.delete(nan_mask_start, -1)
print(nan_mask_start)
print(nan_mask_end)
for start, end in zip(nan_mask_start, nan_mask_end):
    print(start, end)
    start_velocity = yaw_d[start - 2]
    end_velocity = yaw_d[end + 1]
    if yaw[start - 1] > yaw[end + 1]:
        while yaw[start - 1] - yaw[end + 1] > np.pi:
            print(yaw[start - 1] - yaw[end + 1])
            yaw[end + 1 :] += np.pi
    elif yaw[start - 1] < yaw[end + 1]:
        while yaw[start - 1] - yaw[end + 1] > np.pi:
            print(yaw[start - 1] - yaw[end + 1])
            yaw[end + 1 :] -= np.pi
    start = start - int(0.5 / dt)
    end = end + int(0.5 / dt)
    yaw[start : end + 1] = np.linspace(
        yaw[start - 1],
        yaw[end + 1],
        end + 1 - start,
    )

for idx in np.where(mask)[0]:
    adjust = yaw[idx + 1] - yaw[idx]
    yaw[idx + 1 :] -= adjust
yaw_d[:-1] = (yaw[1:] - yaw[:-1]) / dt


final_traj[:, 16] = yaw
final_traj[:, 17] = yaw_d
with open(trajectory_file, "w") as f:
    f.write(traj_header)
    np.savetxt(
        f,
        final_traj,
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
ax.plot(traj[0, :], traj[13, :])

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
ax.set_title("sy")
ax.plot(traj[0, :], traj[14, :])

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
ax.set_title("sz")
ax.plot(traj[0, :], traj[15, :])

fig, axs = plt.subplots(2, 1)
fig.suptitle("Yaw")
ax = axs[0]
ax.grid(True)
ax.set_title("yaw")
ax.plot(traj[0, :], traj[16, :])
ax.scatter(wps[0, :], wps[4, :], color="r")
for change in nan_mask[end_change_mask]:
    ax.axvline(
        x=traj[0, change],
    )
for change in nan_mask[start_change_mask]:
    ax.axvline(x=traj[0, change], color="r")
ax = axs[1]
ax.grid(True)
ax.set_title("yaw rate")
ax.plot(traj[0, :], traj[17, :])
# for change in nan_mask[end_change_mask]:
#     ax.axvline(
#         x=traj[0, change],
#     )
# for change in nan_mask[start_change_mask]:
#     ax.axvline(x=traj[0, change], color="r")

fig, axs = plt.subplots(4, 1)
fig.suptitle("xy")
ax = axs[0]
ax.grid(True)
ax.set_title("xy vel")
ax.plot(traj[0, :], xy_speed)
ax = axs[1]
ax.grid(True)
ax.set_title("xy acc")
ax.plot(traj[0, :], xy_acc)
ax = axs[2]
ax.grid(True)
ax.set_title("xy jerk")
ax.plot(traj[0, :], xy_jerk)
ax = axs[3]
ax.grid(True)
ax.set_title("xy snap")
ax.plot(traj[0, :], xy_snap)


plt.show()
