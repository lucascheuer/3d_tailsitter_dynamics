#pragma once
#include <eigen3/Eigen/Eigen>
#include <sstream>

#include "aircraft_model.hpp"

struct EnvironmentParameters
{
    double gravity;
    double rho;
};

class AircraftDynamics
{
   public:
    static const int kNumStates = 17;

    struct AircraftState
    {
        union
        {
            struct
            {
                double position_x;
                double position_y;
                double position_z;
                double velocity_x;
                double velocity_y;
                double velocity_z;
                double quat_w;
                double quat_x;
                double quat_y;
                double quat_z;
                double omega_x;
                double omega_y;
                double omega_z;
                double elevon_angle_left;
                double elevon_angle_right;
                double motor_omega_left;
                double motor_omega_right;
            };
            struct
            {
                double array[kNumStates];
            };
        };

        friend std::ostream& operator<<(std::ostream& os, const AircraftState& aircraft_state)
        {
            os << aircraft_state.position_x << "," << aircraft_state.position_y << ","
               << aircraft_state.position_z << "," << aircraft_state.velocity_x << ","
               << aircraft_state.velocity_y << "," << aircraft_state.velocity_z << ","
               << aircraft_state.quat_w << "," << aircraft_state.quat_x << ","
               << aircraft_state.quat_y << "," << aircraft_state.quat_z << ","
               << aircraft_state.omega_x << "," << aircraft_state.omega_y << ","
               << aircraft_state.omega_z << "," << aircraft_state.elevon_angle_left << ","
               << aircraft_state.elevon_angle_right << "," << aircraft_state.motor_omega_left << ","
               << aircraft_state.motor_omega_right;
            return os;
        }
    };
    struct AircraftInput
    {
        double elevon_angle_dot_left;
        double elevon_angle_dot_right;
        double motor_omega_dot_left;
        double motor_omega_dot_right;
        friend std::ostream& operator<<(std::ostream& os, const AircraftInput& aircraft_input)
        {
            os << aircraft_input.elevon_angle_dot_left << ","
               << aircraft_input.elevon_angle_dot_right << ","
               << aircraft_input.motor_omega_dot_left << "," << aircraft_input.motor_omega_dot_right
               << std::endl;
            return os;
        }
    };
    AircraftDynamics(AircraftModel params, EnvironmentParameters environmental_params);
    void SetState(AircraftState new_state) { state_ = new_state; }
    AircraftState* GetState() { return &state_; };
    AircraftState* GetStateDot() { return &state_dot_; };
    int Update(double time_step, AircraftInput input);

    static void RkFunctionDerivative(double t, double y[], double yp[]);

    static void WriteForces(std::ostream& os);

   private:
    AircraftModel aircraft_params_;
    EnvironmentParameters environmental_params_;
    AircraftInput input_;
    AircraftState state_;
    AircraftState state_dot_;

    // storage
    double state[kNumStates];
    double state_dot[kNumStates];
};