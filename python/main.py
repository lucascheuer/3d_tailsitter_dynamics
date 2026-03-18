import subprocess
from pathlib import Path, PurePosixPath
from animate import find_data_animate
from plot_output import find_data_plot

'''
Run the simulation. There are two exmaple trajectories "combined" and "simple". Modify the file prefix below to change which set you're using.
Those examples were generates with traj_gen_simple and traj_gen_combined.
'''

# change this to use a specific set of trajectory files
file_prefix = "combined"

sim_path = (
    Path(__file__).resolve().parent.parent
    / "cpp"
    / "sim_and_control"
    / "build"
    / "3d_aircraft_sim"
)
out_folder = Path(__file__).resolve().parent.parent / "out_files"
initial_conditions_folder = Path(__file__).resolve().parent.parent / "initial_conditions"
run_settings_folder = Path(__file__).resolve().parent.parent / "run_settings"
param_folder = Path(__file__).resolve().parent.parent / "param_files"
trajectory_folder = Path(__file__).resolve().parent.parent / "trajectory_files"
waypoint_folder = Path(__file__).resolve().parent.parent / "waypoint_files"

# output files
state_out_file = out_folder / "states.csv"
state_dot_out_file = out_folder / "states_dot.csv"
control_out_file = out_folder / "control.csv"
forces_out_file = out_folder / "forces.csv"

# model and controller input files
aircraft_model_params_file = param_folder / "aircraft_model_params.toml"
controller_params_file = param_folder / "controller_params.toml"

# trajectory files
run_params_file = run_settings_folder / PurePosixPath(file_prefix + "_run_settings.toml")
initial_condition_params_file = initial_conditions_folder / PurePosixPath(file_prefix + "_initial_conditions.toml")
trajectory_file = trajectory_folder / PurePosixPath(file_prefix + "_trajectory.csv")
waypoint_file = waypoint_folder / PurePosixPath(file_prefix + "_waypoints.csv")


input_list = [
    sim_path,
    state_out_file,
    state_dot_out_file,
    control_out_file,
    forces_out_file,
    aircraft_model_params_file,
    controller_params_file,
    run_params_file,
    initial_condition_params_file,
    trajectory_file,
]

subprocess.run(input_list)

find_data_animate(
    state_out_file,
    forces_out_file,
    control_out_file,
    aircraft_model_params_file,
    run_params_file,
    waypoint_file,
)

find_data_plot(state_out_file, state_dot_out_file, control_out_file)
