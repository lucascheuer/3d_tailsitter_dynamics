#pragma once
#include <eigen3/Eigen/Eigen>
#include <iostream>

#include "toml.hpp"

struct AircraftModel
{
   public:
    double mass;
    Eigen::Matrix3d moment_of_inertia = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d moment_of_inertia_inv = Eigen::Matrix3d::Zero();
    double prop_thrust_coeff;
    double prop_moment_coeff;
    double prop_max_omega;
    double prop_max_omega_dot;
    double wing_surface_area;
    double propeller_disc_area;
    double minimum_drag_coeff;
    double maximum_drag_coeff;
    double roll_rate_drag_coeff;
    double pitch_rate_drag_coeff;
    double yaw_rate_drag_coeff;
    Eigen::Vector3d linear_elevon_effectiveness;
    Eigen::Vector3d rotational_elevon_effectiveness;
    double max_elevon_angle;
    double max_elevon_angle_dot;
    double wingspan;
    double chord;
    double elevon_percentage;
    double delta_r;
    Eigen::Vector3d propeller_moment_arm_left;
    Eigen::Vector3d propeller_moment_arm_right;
    Eigen::Vector3d aerodynamic_moment_arm_left;
    Eigen::Vector3d aerodynamic_moment_arm_right;
    Eigen::Matrix3d phi_m_omega = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d phi_f_v = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d phi_m_v = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d B = Eigen::Matrix3d::Zero();
    AircraftModel(std::string filename);
};