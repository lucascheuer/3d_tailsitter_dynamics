#include <chrono>
#include <iostream>

#include "aircraft_controller.hpp"
#include "aircraft_controller_params.hpp"
#include "aircraft_dynamics.hpp"
#include "aircraft_model.hpp"
#include "helpers.hpp"
#include "toml.hpp"

AircraftDynamics::AircraftState ParseInitialConditions(std::string filename)
{
    AircraftDynamics::AircraftState initial_conditions;
    toml::table tbl;
    try
    {
        tbl = toml::parse_file(filename);
        using namespace TomlParseHelpers;
        initial_conditions.position_x = ParseDouble(tbl, "pos_x");
        initial_conditions.position_y = ParseDouble(tbl, "pos_y");
        initial_conditions.position_z = ParseDouble(tbl, "pos_z");

        double roll = ParseDouble(tbl, "roll") / 180.0 * M_PI;
        double pitch = ParseDouble(tbl, "pitch") / 180.0 * M_PI;
        double yaw = ParseDouble(tbl, "yaw") / 180.0 * M_PI;
        Eigen::Quaterniond q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                               Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());
        double vel_x_inertial = ParseDouble(tbl, "vel_x");
        double vel_y_inertial = ParseDouble(tbl, "vel_y");
        double vel_z_inertial = ParseDouble(tbl, "vel_z");
        Eigen::Vector3d velocity_body =
            q.conjugate() * Eigen::Vector3d(vel_x_inertial, vel_y_inertial, vel_z_inertial);
        std::cout << velocity_body << std::endl;

        initial_conditions.velocity_x = velocity_body.x();
        initial_conditions.velocity_y = velocity_body.y();
        initial_conditions.velocity_z = velocity_body.z();
        initial_conditions.quat_w = q.w();
        initial_conditions.quat_x = q.x();
        initial_conditions.quat_y = q.y();
        initial_conditions.quat_z = q.z();
        initial_conditions.omega_x = ParseDouble(tbl, "omega_x");
        initial_conditions.omega_y = ParseDouble(tbl, "omega_y");
        initial_conditions.omega_z = ParseDouble(tbl, "omega_z");
        initial_conditions.elevon_angle_left = ParseDouble(tbl, "elevon_left");
        initial_conditions.elevon_angle_right = ParseDouble(tbl, "elevon_right");
        initial_conditions.motor_omega_left = ParseDouble(tbl, "motor_omega_left");
        initial_conditions.motor_omega_right = ParseDouble(tbl, "motor_omega_right");
        std::cout << initial_conditions.motor_omega_left << "\t"
                  << initial_conditions.motor_omega_right << std::endl;

    } catch (const toml::parse_error& err)
    {
        std::cerr << "Parsing failed:\n" << err << "\n";
    }
    return initial_conditions;
}

int OpenTrajFile(std::ifstream& traj_file, std::string traj_file_name)
{
    traj_file.open(traj_file_name);
    if (!traj_file.is_open())
    {
        return -1;
    } else
    {
        std::string line;
        std::getline(traj_file, line);
        return 0;
    }
}

Eigen::Vector3d ParseXYZ(std::stringstream& traj_line)
{
    Eigen::Vector3d to_ret = Eigen::Vector3d::Zero();
    for (int ii = 0; ii < 3; ++ii)
    {
        std::string field;
        std::getline(traj_line, field, ',');
        to_ret[ii] = std::stod(field);
    }
    return to_ret;
}

bool ParseTrajRow(
    std::ifstream& traj_file,
    TrackingController::AircraftControllerDesired& controller_desired)
{
    // times,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,jerk_x,jerk_y,jerk_z,yaw,yaw_dot
    TrackingController::AircraftControllerDesired to_ret;

    std::string line;
    if (!std::getline(traj_file, line))
    {
        return false;
    }
    std::stringstream line_stream(line);
    std::string field;
    try
    {
        std::getline(line_stream, field, ',');
        controller_desired.pos_des = ParseXYZ(line_stream);
        controller_desired.vel_des = ParseXYZ(line_stream);
        controller_desired.acc_des = ParseXYZ(line_stream);
        controller_desired.jerk_des = ParseXYZ(line_stream);
        std::getline(line_stream, field, ',');
        controller_desired.yaw_des = std::stod(field);
        std::getline(line_stream, field, ',');
        controller_desired.yaw_rate_des = std::stod(field);
    } catch (const std::out_of_range& oor)
    {
        std::cout << "out of range error" << std::endl;
        return false;
    } catch (const std::invalid_argument& ia)
    {
        std::cout << "invalid argument error" << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char* argv[])
{
    // get a filename
    std::string output_filename = "out.csv";
    if (argc > 1)
    {
        output_filename = std::string(argv[1]);
    }
    std::string output_dot_filename = "out_dot.csv";
    if (argc > 2)
    {
        output_dot_filename = std::string(argv[2]);
    }
    std::string control_filename = "control.csv";
    if (argc > 3)
    {
        control_filename = std::string(argv[3]);
    }
    std::string forces_filename = "forces.csv";
    if (argc > 4)
    {
        forces_filename = std::string(argv[4]);
    }
    std::string aircraft_model_params = "aircraft_model_params.toml";
    if (argc > 5)
    {
        aircraft_model_params = std::string(argv[5]);
    }
    std::string controller_params_filename = "controller_params.toml";
    if (argc > 6)
    {
        controller_params_filename = std::string(argv[6]);
    }
    std::string run_params = "run_params.toml";
    if (argc > 7)
    {
        run_params = std::string(argv[7]);
    }
    std::string initial_conditions_filename = "initial_conditions.toml";
    if (argc > 8)
    {
        initial_conditions_filename = std::string(argv[8]);
    }
    std::string trajectory_filename = "trajectory.csv";
    if (argc > 9)
    {
        trajectory_filename = std::string(argv[9]);
    }

    // get run parameters
    double t_start = 0.0;
    double t_end;
    double t_step;
    toml::table tbl;
    try
    {
        tbl = toml::parse_file(run_params);
        using namespace TomlParseHelpers;
        t_end = ParseDouble(tbl, "run_time");
        t_step = ParseDouble(tbl, "time_step");
    } catch (const toml::parse_error& err)
    {
        std::cerr << "Parsing failed:\n" << err << "\n";
        return 0;
    }

    std::ofstream output_file(output_filename);
    if (output_file.is_open())
    {
        output_file << "t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z,omega_x,"
                       "omega_y,omega_z,elevon_left,elevon_right,motor_omega_left,motor_omega_right"
                    << std::endl;
    } else
    {
        std::cerr << "Error: Unable to open file output file " << output_filename << std::endl;
        return 0;
    }
    std::ofstream output_dot_file(output_dot_filename);
    if (output_dot_file.is_open())
    {
        output_dot_file
            << "t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z,omega_x,"
               "omega_y,omega_z,elevon_left,elevon_right,motor_omega_left,motor_omega_right"
            << std::endl;
    } else
    {
        std::cerr << "Error: Unable to open file output dot file " << output_dot_filename
                  << std::endl;
        return 0;
    }
    std::ofstream forces(forces_filename);
    if (forces.is_open())
    {
    } else
    {
        std::cerr << "Error: Unable to open file forces file " << forces_filename << std::endl;
        return 0;
    }

    AircraftModel aircraft_params(aircraft_model_params);
    EnvironmentParameters environmental_params = {.gravity = 9.81, .rho = 1.225};
    AircraftDynamics dynamics(aircraft_params, environmental_params);
    AircraftControllerParameters controller_params =
        AircraftControllerParameters::GetControllerParameters(
            controller_params_filename, aircraft_params, environmental_params);
    TrackingController controller(controller_params, control_filename);
    AircraftDynamics::AircraftInput input{
        .elevon_angle_dot_left = 0,
        .elevon_angle_dot_right = 0,
        .motor_omega_dot_left = 0,
        .motor_omega_dot_right = 0};
    std::ifstream traj_file;
    bool use_traj = false;
    AircraftDynamics::AircraftState initial_conditions;
    if (!OpenTrajFile(traj_file, trajectory_filename))
    {
        use_traj = true;
    }
    initial_conditions = ParseInitialConditions(initial_conditions_filename);
    dynamics.SetState(initial_conditions);
    output_file << 0.0 << "," << *dynamics.GetState() << "\n";

    double vel_x_last = 0;
    double vel_y_last = 0;
    double vel_z_last = 0;
    double omega_x_last = 0;
    double omega_y_last = 0;
    double omega_z_last = 0;
    int step_count = (t_end - t_start) / t_step;
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < step_count; ++step)
    {
        // std::cout << "step: " << step << "/" << step_count << std::endl;
        if (dynamics.Update(t_step, input) > 0)
        {
            output_file << t_step * step << "," << *dynamics.GetState() << "\n";
            forces << t_step * step << ",";
            dynamics.WriteForces(forces);
            forces << "\n";
            // double acc_x = dynamics.GetStateDot()->velocity_x;
            // double acc_y = dynamics.GetStateDot()->velocity_y;
            // double acc_z = dynamics.GetStateDot()->velocity_z;
            // double omega_dot_x = dynamics.GetStateDot()->omega_x;
            // double omega_dot_y = dynamics.GetStateDot()->omega_x;
            // double omega_dot_z = dynamics.GetStateDot()->omega_x;
            double acc_x = (dynamics.GetState()->velocity_x - vel_x_last) / t_step;
            double acc_y = (dynamics.GetState()->velocity_y - vel_y_last) / t_step;
            double acc_z = (dynamics.GetState()->velocity_z - vel_z_last) / t_step;
            double omega_dot_x = (dynamics.GetState()->omega_x - omega_x_last) / t_step;
            double omega_dot_y = (dynamics.GetState()->omega_y - omega_y_last) / t_step;
            double omega_dot_z = (dynamics.GetState()->omega_z - omega_z_last) / t_step;

            vel_x_last = dynamics.GetState()->velocity_x;
            vel_y_last = dynamics.GetState()->velocity_y;
            vel_z_last = dynamics.GetState()->velocity_z;
            omega_x_last = dynamics.GetState()->omega_x;
            omega_y_last = dynamics.GetState()->omega_y;
            omega_z_last = dynamics.GetState()->omega_z;
            // std::cout << step << "\t";
            // double time = t_step * step;
            // double amplitude = 1.0;
            // Eigen::Vector3d pos_des(amplitude * sin(time * M_PI * 0.5), 0, 0);
            // Eigen::Vector3d vel_des(amplitude * 0.5 * M_PI * cos(time * M_PI * 0.5), 0, 0);
            // Eigen::Vector3d acc_des(-amplitude * 0.25 * M_PI * M_PI * sin(time * M_PI * 0.5), 0,
            // 0); Eigen::Vector3d jerk_des(
            //     -amplitude * 0.125 * M_PI * M_PI * M_PI * cos(time * M_PI * 0.5), 0, 0);

            if (use_traj)
            {
                TrackingController::AircraftControllerDesired current_step_des;
                use_traj = ParseTrajRow(traj_file, current_step_des);
                input = controller.Update(
                    current_step_des,
                    *dynamics.GetState(),
                    input,
                    Eigen::Vector3d(acc_x, acc_y, acc_z),
                    Eigen::Vector3d(omega_dot_x, omega_dot_y, omega_dot_z));
            } else
            {
                input = controller.Update(
                    *dynamics.GetState(),
                    input,
                    Eigen::Vector3d(acc_x, acc_y, acc_z),
                    Eigen::Vector3d(omega_dot_x, omega_dot_y, omega_dot_z));
            }
            // std::cout << std::endl;
        } else
        {
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Function execution time: " << duration.count() << " seconds" << std::endl;
    output_file.close();
    traj_file.close();
}