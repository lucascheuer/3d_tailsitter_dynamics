#pragma once
#include <eigen3/Eigen/Eigen>
#include <iostream>

#include "aircraft_model.hpp"
#include "toml.hpp"

struct AircraftControllerModel
{
   public:
    double mass;
    Eigen::Matrix3d moment_of_inertia = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d moment_of_inertia_inv = Eigen::Matrix3d::Zero();
    double prop_thrust_coeff;
    double prop_moment_coeff;
    double prop_max_omega;
    double wing_lift_coeff;             // C_l_v
    double wing_drag_coeff;             // C_d_v
    double wing_lift_coeff_thrust;      // C_t // THIS IS WRONG FIX
    double wing_drag_coeff_thrust;      // C_d_t
    double elevon_lift_coeff_airspeed;  // delta_C_l_v
    double elevon_lift_coeff_thrust;    // delta_C_l_t
    double pitch_coeff_thrust;          // C_u_T
    double max_elevon_angle;

    double thrust_l_y;
    double flap_length_x;
    double flap_length_y;
    AircraftControllerModel();
    AircraftControllerModel(
        AircraftModel full_params,
        double elevon_lift_coeff_airspeed_input,
        double elevon_lift_coeff_thrust_input,
        double pitch_coeff_thrust_input);
};