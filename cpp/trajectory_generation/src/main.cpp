#include <fstream>
#include <iomanip>
#include <iostream>

#include "min_snap_traj.hpp"
#include "state.hpp"

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cout << "No waypoint arguments" << std::endl;
        exit(0);
    }
    std::string trajectory_file_name = std::string(argv[1]);
    std::ofstream trajectory_file(trajectory_file_name);
    if (trajectory_file.is_open())
    {
        trajectory_file
            << "t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z,omega_x,"
               "omega_y,omega_z,elevon_left,elevon_right,motor_omega_left,motor_omega_right"
            << std::endl;
        trajectory_file << std::fixed;
        trajectory_file << std::setprecision(5);
    } else
    {
        std::cout << "Failed to Open Trajectory File" << std::endl;
        exit(0);
    }
    MinSnapTraj traj;
    float control_frequency = std::stod(argv[2]);
    float traj_speed = std::stod(argv[3]);
    int num_waypoints = std::stoi(argv[4]);
    int waypoint_base_arg = 5;
    for (int waypoint_num = 0; waypoint_num < num_waypoints; ++waypoint_num)
    {
        int current_waypoint_base_count = waypoint_base_arg + waypoint_num * 4;
        Eigen::Vector3d waypoint_position;
        waypoint_position.x() = std::stod(argv[current_waypoint_base_count + 0]);
        waypoint_position.y() = std::stod(argv[current_waypoint_base_count + 1]);
        waypoint_position.z() = std::stod(argv[current_waypoint_base_count + 2]);
        std::cout << "waypoint #" << waypoint_num << "\n" << waypoint_position << "\n\n";
        double yaw = std::stod(argv[current_waypoint_base_count + 3]);
        MinSnapTraj::Waypoint new_waypoint(waypoint_position, yaw);
        traj.AddWaypoint(new_waypoint);
    }

    bool solved = traj.Solve(traj_speed);
    if (solved && traj.solved())
    {
        std::cout << "traj solved" << std::endl;
    }
    if (!solved)
    {
        std::cout << "traj failed to solve" << std::endl;
    }
    std::cout << traj.GetWaypointCount() << std::endl;
    double start_time = 0;
    double end_time = traj.EndTime();
    double dt = 1.0 / control_frequency;
    for (double time = start_time; time < end_time; time += dt)
    {
        State state;
        traj.Evaluate(time, state);
        trajectory_file << time << "," << state.x << "," << state.y << "," << state.z << ","
                        << state.vx << "," << state.vy << "," << state.vz << "," << state.ax << ","
                        << state.ay << "," << state.az << "," << state.jx << "," << state.jy << ","
                        << state.jz << "," << state.yaw << "," << state.vyaw << "\n";
    }
    trajectory_file.close();
    std::cout << "Traj Length = " << end_time << std::endl;
    return 0;
}