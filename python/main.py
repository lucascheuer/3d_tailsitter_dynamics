import subprocess
from pathlib import Path

sim_path = Path(__file__).resolve().parent.parent / "cpp" / "build" / "3d_aircraft_sim"
out_folder = Path(__file__).resolve().parent.parent / "out"
state_out_file = out_folder / "states.csv"
state_dot_out_file = out_folder / "states_dot.csv"
control_out_file = out_folder / "control.csv"

subprocess.run([sim_path, state_out_file, state_dot_out_file, control_out_file])
