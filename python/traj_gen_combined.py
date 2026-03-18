from pathlib import Path
import numpy as np
from traj_gen_circle import generate_circle
from traj_gen_min_snap import generate_minsnap


''' 
This file will generate a trajectory and waypoint file at at trajectory_files/combined_trajectory.csv
and waypoint_files/combined_waypoints.csv respectively.

The trajectory combines a minsnap trajectory with a circle trajectory and then another minsnap trajectory.
'''

out_folder = Path(__file__).resolve().parent.parent / "trajectory_files"
waypoint_folder = Path(__file__).resolve().parent.parent / "waypoint_files"
initial_conditions_folder = Path(__file__).resolve().parent.parent / "initial_conditions"
run_settings_folder = Path(__file__).resolve().parent.parent / "run_settings"

trajectory_file = out_folder / "combined_trajectory.csv"
waypoint_file = waypoint_folder / "combined_waypoints.csv"
initial_conditions_file = initial_conditions_folder / "combined_initial_conditions.toml"
run_settings_file = run_settings_folder / "combined_run_settings.toml"


controller_frequency = 1000
traj_speed = 16

total_run_time = 0
time_step = 1.0 / controller_frequency

waypoints = [
    [0, 0, 0, np.pi / 2],
    [0, 8, 0, np.pi / 2],
    [0, 16, 0, 0],
]

# get the initial conditions from the initial waypoints
initial_conditions = """pos_x = {:.5f}
pos_y = {:.5f}
pos_z = {:.5f}
vel_x_body = 0
vel_y_body = 0
vel_z_body = 0
roll = 0
pitch = 90
yaw = {:.5f}
omega_x = 0
omega_y = 0
omega_z = 0
elevon_left = 0
elevon_right = 0
motor_omega_left = -285
motor_omega_right = 285""".format(waypoints[0][0], waypoints[0][1], waypoints[0][2], np.rad2deg(waypoints[0][3]))

with open(initial_conditions_file, "w") as f:
    f.write(initial_conditions)

start_point_derivatives = [
    [0, 0, 0, 0],  # vel
    [0, 0, 0, 0],  # accl
    [0, 0, 0, 0],  # jerk
    [0, 0, 0, 0],  # snap
]

# generate the circle trajectory
circle_freq = 1.0 / 10.0
circle_diameter = 10.0
circle_time_len = 1 / circle_freq
circle_traj = generate_circle(
    traj_speed,
    circle_time_len,
    circle_freq,
    circle_diameter,
    waypoints[-1][0],
    waypoints[-1][1],
    z_offset=waypoints[-1][2],
    spiral_speed=-0.0,
)

total_run_time += circle_time_len # add the circle traj run time to the total run time

# set the endpoints derivatives of the first minsnap trajectory to be the same as the startpoints of the circle trajectory
end_point_derivatives = np.zeros((4, 4))
end_point_derivatives[0, 0:3] = circle_traj[0, 4:7]
end_point_derivatives[1, 0:3] = circle_traj[0, 7:10]
end_point_derivatives[2, 0:3] = circle_traj[0, 10:13]
end_point_derivatives[3, 0:3] = circle_traj[0, 13:16]
end_point_derivatives[0, 3] = circle_traj[0, 17]

# generate the first minsnap trajectory
minsnap_traj_one, wp_one = generate_minsnap(
    waypoints, traj_speed, start_point_derivatives, end_point_derivatives
)

total_run_time += traj_speed # add the first minsnap traj to the total run time

# set up the new waypoints for the second minsnap trajectory
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

# set the start point derivatives to match the end of the circle trajectory
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

# generate the second minsnap trajectory
minsnap_traj_two, wp_two = generate_minsnap(
    waypoints, traj_speed, start_point_derivatives, end_point_derivatives
)

total_run_time += traj_speed # add the second minsnap traj to the total run time

# set up the run settings
run_time_text = "run_time = " + str(total_run_time) + "\n"
time_step_text = "time_step = " + str(time_step)

with open(run_settings_file, "w") as f:
    f.write(run_time_text)
    f.write(time_step_text)

# modify the start
minsnap_traj_two_start_time = circle_traj[-1, 0]
minsnap_traj_two[:, 0] += minsnap_traj_two_start_time
wp_two[:, 0] += minsnap_traj_two_start_time

traj_header = "times,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,jerk_x,jerk_y,jerk_z,snap_x,snap_y,snap_z,yaw,yaw_dot\n"
wp_header = "t,pos_x,pos_y,pos_z,yaw\n"

# combine trajectories and waypoints
final_traj = np.concatenate((minsnap_traj_one, circle_traj, minsnap_traj_two), axis=0)
final_wp = np.concatenate((wp_one, wp_two), axis=0)

# write out the files
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
    final_wp
    np.savetxt(
        f,
        final_wp,
        fmt="%.5f",
        delimiter=",",
        comments="",
    )
