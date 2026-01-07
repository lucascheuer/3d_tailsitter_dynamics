import subprocess
from pathlib import Path
import numpy as np
import csv


def generate_minsnap(
    waypoints,
    traj_speed,
    start_point_derivatives=None,
    end_point_derivatives=None,
    descent_rate=0.0001,
    step_limit=100,
    speed_weight=0,
    controller_frequency=1000,
):
    traj_gen_path = (
        Path(__file__).resolve().parent.parent
        / "cpp"
        / "trajectory_generation"
        / "build"
        / "min_snap_generator"
    )
    out_folder = Path("/tmp")

    trajectory_file = out_folder / "min_snap.csv"
    waypoint_file = out_folder / "min_snap_waypoint.csv"

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
    if not (start_point_derivatives is None or end_point_derivatives is None):
        print("derivatives being used")
        for derivative in start_point_derivatives:
            for item in derivative:
                input_list.append(str(item))
        for derivative in end_point_derivatives:
            for item in derivative:
                input_list.append(str(item))

    subprocess.run(input_list, capture_output=False, text=True)

    with open(trajectory_file, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        traj = []
        next(csv_reader)
        for row in csv_reader:
            float_row = [float(item) for item in row]
            traj.append(float_row)

        traj = np.array(traj)
    with open(waypoint_file, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        wps = []
        next(csv_reader)
        for row in csv_reader:
            float_row = [float(item) for item in row]
            wps.append(float_row)

        wps = np.array(wps)
    return traj, wps
