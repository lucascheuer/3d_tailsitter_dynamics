import subprocess
from pathlib import Path
from animate import find_data_animate
from plot_output import find_data_plot

sim_path = (
    Path(__file__).resolve().parent.parent
    / "cpp"
    / "sim_and_control"
    / "build"
    / "3d_aircraft_sim"
)
out_folder = Path(__file__).resolve().parent.parent / "out_files"
param_folder = Path(__file__).resolve().parent.parent / "param_files"
trajectory_folder = Path(__file__).resolve().parent.parent / "trajectory_files"

# output files
state_out_file = out_folder / "states.csv"
state_dot_out_file = out_folder / "states_dot.csv"
control_out_file = out_folder / "control.csv"
forces_out_file = out_folder / "forces.csv"

# input files
aircraft_model_params_file = param_folder / "aircraft_model_params.toml"
controller_params_file = param_folder / "controller_params.toml"
run_params_file = param_folder / "run_settings.toml"
initial_condition_params_file = param_folder / "initial_conditions.toml"
trajectory_file = trajectory_folder / "trajectory.csv"

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
)

find_data_plot(state_out_file, state_dot_out_file, control_out_file)
