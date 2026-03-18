from pathlib import Path
import numpy as np
from traj_gen_min_snap import generate_minsnap


''' 
This file will generate a trajectory and waypoint file at at trajectory_files/simple_trajectory.csv
and waypoint_files/simple_waypoints.csv respectively.
'''

out_folder = Path(__file__).resolve().parent.parent / "trajectory_files"
waypoint_folder = Path(__file__).resolve().parent.parent / "waypoint_files"
initial_conditions_folder = Path(__file__).resolve().parent.parent / "initial_conditions"
run_settings_folder = Path(__file__).resolve().parent.parent / "run_settings"

trajectory_file = out_folder / "simple_trajectory.csv"
waypoint_file = waypoint_folder / "simple_waypoints.csv"
initial_conditions_file = initial_conditions_folder / "simple_initial_conditions.toml"
run_settings_file = run_settings_folder / "simple_run_settings.toml"

controller_frequency = 1000
traj_speed = 16

total_run_time = traj_speed
time_step = 1.0 / controller_frequency

run_time_text = "run_time = " + str(total_run_time) + "\n"
time_step_text = "time_step = " + str(time_step)

with open(run_settings_file, "w") as f:
    f.write(run_time_text)
    f.write(time_step_text)



waypoints = [
    [0, 0, 0, 0],
    [0, 0, 0, np.pi / 2],
    [0, 8, 0, np.pi / 2],
    [0, 16, 0, 0],
]

initial_conditions = """pos_x = {:.5f}
pos_y = {:.5f}
pos_z = {:.5f}
vel_x_body = 0
vel_y_body = 0
vel_z_body = 0
roll = 0
pitch = 90
yaw = 0
omega_x = 0
omega_y = 0
omega_z = 0
elevon_left = 0
elevon_right = 0
motor_omega_left = -285
motor_omega_right = 285""".format(waypoints[0][0], waypoints[0][1], waypoints[0][2])

with open(initial_conditions_file, "w") as f:
    f.write(initial_conditions)



start_point_derivatives = [
    [0, 0, 0, 0],  # vel
    [0, 0, 0, 0],  # accl
    [0, 0, 0, 0],  # jerk
    [0, 0, 0, 0],  # snap
]

end_point_derivatives = [
    [0, 0, 0, 0],  # vel
    [0, 0, 0, 0],  # accl
    [0, 0, 0, 0],  # jerk
    [0, 0, 0, 0],  # snap
]

minsnap_traj, wp = generate_minsnap(
    waypoints, traj_speed, start_point_derivatives, end_point_derivatives
)
traj_header = "times,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,jerk_x,jerk_y,jerk_z,snap_x,snap_y,snap_z,yaw,yaw_dot\n"
wp_header = "t,pos_x,pos_y,pos_z,yaw\n"

with open(trajectory_file, "w") as f:
    f.write(traj_header)
    np.savetxt(
        f,
        minsnap_traj,
        fmt="%.5f",
        delimiter=",",
        comments="",
    )
    
with open(waypoint_file, "w") as f:
    f.write(wp_header)
    np.savetxt(
        f,
        wp,
        fmt="%.5f",
        delimiter=",",
        comments="",
    )
