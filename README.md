Attempting to implement the controller from "Global Incremental Flight Control for Agile Maneuvering of a Tailsitter Flying Wing" from Ezra Tal and Sertac Karaman.

Simulation is an implementation of "Global Singularity-Free Aerodynamic Model for Algorithmic Flight Control of Tail Sitters" from Leandro Ribeiro Lustosa, François Defaÿ, Jean-Marc Moschetta

To run, install the python deps with UV, and build the sim and traj gen stuff by going into the cpp/sim_and_control or cpp/trajectory_generation folders, making a build dir, and building. Both require eigen, and the traj gen requires osqp eigen.

In the python directory run main.py to simulate, or run traj_gen_simple.py or traj_gen_combined.py to generate trajectory files to run.

```
cd cpp/sim_and_control
mkdir build
cd build
cmake ..
make -j$(nproc)
```