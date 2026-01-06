#include <fstream>
#include <iomanip>
#include <iostream>

#include "min_snap_traj.hpp"
#include "state.hpp"

int main(int argc, char* argv[])
{
    if (argc < 11)
    {
        std::cout << "No waypoint arguments" << std::endl;
        exit(0);
    }
    int append = std::stoi(argv[4]);
    // std::cout << "Appending to Files: " << append << std::endl;

    std::string trajectory_file_name = std::string(argv[1]);
    std::ofstream trajectory_file;
    if (append)
    {
        trajectory_file.open(trajectory_file_name, std::ios::app);
    } else
    {
        trajectory_file.open(trajectory_file_name);
    }
    if (trajectory_file.is_open())
    {
        if (!append)
        {
            trajectory_file << "t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,jerk_x,"
                               "jerk_y,jerk_z,yaw,yaw_dot"
                            << std::endl;
        }
        trajectory_file << std::fixed;
        trajectory_file << std::setprecision(5);
    } else
    {
        std::cout << "Failed to Open Trajectory File" << std::endl;
        exit(0);
    }
    std::string waypoint_file_name = std::string(argv[2]);

    std::ofstream waypoint_file;
    if (append)
    {
        waypoint_file.open(waypoint_file_name, std::ios::app);
    } else
    {
        waypoint_file.open(waypoint_file_name);
    }
    if (waypoint_file.is_open())
    {
        if (!append)
        {
            waypoint_file << "t,pos_x,pos_y,pos_z,yaw" << std::endl;
        }
        waypoint_file << std::fixed;
        waypoint_file << std::setprecision(5);
    } else
    {
        std::cout << "Failed to Open Waypoint File" << std::endl;
        exit(0);
    }
    MinSnapTraj traj;
    float control_frequency = std::stod(argv[3]);
    float traj_speed = std::stod(argv[5]);
    float speed_weight = std::stod(argv[6]);
    float descent_rate = std::stod(argv[7]);
    int step_limit = std::stoi(argv[8]);
    int num_waypoints = std::stoi(argv[9]);
    int waypoint_base_arg = 10;
    int derivatives_base_arg = 0;
    for (int waypoint_num = 0; waypoint_num < num_waypoints; ++waypoint_num)
    {
        int current_waypoint_base_count = waypoint_base_arg + waypoint_num * 4;
        Eigen::Vector3d waypoint_position;
        waypoint_position.x() = std::stod(argv[current_waypoint_base_count + 0]);
        waypoint_position.y() = std::stod(argv[current_waypoint_base_count + 1]);
        waypoint_position.z() = std::stod(argv[current_waypoint_base_count + 2]);
        // std::cout << "waypoint #" << waypoint_num << "\n" << waypoint_position << "\n\n";
        double yaw = std::stod(argv[current_waypoint_base_count + 3]);
        derivatives_base_arg = current_waypoint_base_count + 3;
        MinSnapTraj::Waypoint new_waypoint(waypoint_position, yaw);
        traj.AddWaypoint(new_waypoint);
    }
    derivatives_base_arg += 1;
    std::vector<MinSnapTraj::Waypoint> start_point_derivatives;
    for (int derivative_num = 0; derivative_num < 4; ++derivative_num)
    {
        int current_waypoint_base_count = derivatives_base_arg + derivative_num * 4;
        Eigen::Vector3d waypoint_position;
        waypoint_position.x() = std::stod(argv[current_waypoint_base_count + 0]);
        waypoint_position.y() = std::stod(argv[current_waypoint_base_count + 1]);
        waypoint_position.z() = std::stod(argv[current_waypoint_base_count + 2]);
        double yaw = std::stod(argv[current_waypoint_base_count + 3]);
        MinSnapTraj::Waypoint new_waypoint(waypoint_position, yaw);
        start_point_derivatives.push_back(new_waypoint);
    }
    std::vector<MinSnapTraj::Waypoint> end_point_derivatives;
    for (int derivative_num = 0; derivative_num < 4; ++derivative_num)
    {
        int current_waypoint_base_count = derivatives_base_arg + 16 + derivative_num * 4;
        Eigen::Vector3d waypoint_position;
        waypoint_position.x() = std::stod(argv[current_waypoint_base_count + 0]);
        waypoint_position.y() = std::stod(argv[current_waypoint_base_count + 1]);
        waypoint_position.z() = std::stod(argv[current_waypoint_base_count + 2]);
        double yaw = std::stod(argv[current_waypoint_base_count + 3]);
        MinSnapTraj::Waypoint new_waypoint(waypoint_position, yaw);
        end_point_derivatives.push_back(new_waypoint);
    }
    traj.SetStartDerivatives(start_point_derivatives);
    traj.SetEndDerivatives(end_point_derivatives);
    bool solved = traj.Solve(speed_weight, traj_speed, step_limit, descent_rate);
    if (solved && traj.solved())
    {
        // std::cout << "traj solved" << std::endl;
    }
    if (!solved)
    {
        std::cout << "traj failed to solve" << std::endl;
    }
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
    double time = 0;
    for (int waypoint = 0; waypoint < traj.GetWaypointCount(); ++waypoint)
    {
        waypoint_file << time << "," << *traj.GetWaypoint(waypoint) << "\n";
        if (waypoint < traj.GetWaypointCount() - 1)
        {
            time += traj.GetTimes()[waypoint];
        }
    }
    waypoint_file.close();
    return 0;
}