#pragma once
#include <eigen3/Eigen/Eigen>
#include <iostream>

#include "aircraft_controller_model.hpp"
#include "aircraft_dynamics.hpp"
#include "aircraft_model.hpp"
#include "helpers.hpp"

struct AircraftControllerParameters
{
   public:
    AircraftControllerModel model_params;
    EnvironmentParameters environment_params;
    Eigen::Matrix3d pos_gain;
    Eigen::Matrix3d vel_gain;
    Eigen::Matrix3d acc_gain;
    Eigen::Matrix3d quat_gain;
    Eigen::Matrix3d omega_gain;
    double controller_frequency;

    static AircraftControllerParameters GetControllerParameters(
        std::string param_file,
        AircraftModel full_params,
        EnvironmentParameters environment_params);
};
