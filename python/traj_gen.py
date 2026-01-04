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

# output files
trajectory_file = out_folder / "new_traj.csv"

controller_frequency = 1000

waypoints = [
    [0, 0, 0, 0],
    [5, 0, 0, np.pi / 2],
    [10, 0, 0, 0],
]

input_list = [
    traj_gen_path,
    trajectory_file,
    str(controller_frequency),
    str(1.0),
    str(len(waypoints)),
]
for waypoint in waypoints:
    for item in waypoint:
        input_list.append(str(item))
print(input_list)

subprocess.run(input_list)
