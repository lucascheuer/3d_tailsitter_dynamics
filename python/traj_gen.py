import subprocess
from pathlib import Path
import numpy as np

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
trajectory_file = out_folder / "trajectory.csv"
waypoint_file = waypoint_folder / "waypoints.csv"

controller_frequency = 1000
traj_speed = 35
speed_weight = 0
descent_rate = 0.00000001
step_limit = 1000

angle_step = np.pi / 8
diameter = 16
lateral_offset = 0.1
angles = np.arange(0, np.pi * 2 + angle_step, angle_step)
xs = 0.5 * diameter * np.cos(angles - np.pi / 2) + lateral_offset
ys = 0.5 * diameter * np.sin(angles - np.pi / 2) + diameter * 0.5
waypoints = [[0, 0, 0, 0]]
for x, y, angle in zip(xs, ys, angles):
    waypoints.append([x, y, 0, angle])
waypoints.append([lateral_offset + lateral_offset, 0, 0, np.pi * 4.0])


input_list = [
    traj_gen_path,
    trajectory_file,
    waypoint_file,
    str(controller_frequency),
    str(traj_speed),
    str(speed_weight),
    str(descent_rate),
    str(step_limit),
    str(len(waypoints)),
]
for waypoint in waypoints:
    for item in waypoint:
        input_list.append(str(item))
print(input_list)

subprocess.run(input_list)
