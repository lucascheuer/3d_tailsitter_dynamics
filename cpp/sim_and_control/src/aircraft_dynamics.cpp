#include "aircraft_dynamics.hpp"

#include <math.h>

#include <eigen3/Eigen/Eigen>

#include "rkf45.hpp"

static AircraftModel* aircraft_rk = nullptr;
static EnvironmentParameters* environment_rk = nullptr;
static AircraftDynamics::AircraftInput input_rk;

template <typename T>
int signum(T val)
{
    if (val > 0)
        return 1;
    if (val < 0)
        return -1;
    return 0;  // val is 0
}
Eigen::Matrix3d matrix_cross(Eigen::Vector3d vector)
{
    Eigen::Matrix3d ret_mat;
    ret_mat << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1), vector(0), 0;
    return ret_mat;
}

AircraftDynamics::AircraftDynamics(
    AircraftModel aircraft_params,
    EnvironmentParameters environmental_params)
    : aircraft_params_(aircraft_params), environmental_params_(environmental_params)
{
    aircraft_rk = &aircraft_params_;
    environment_rk = &environmental_params_;
}

int AircraftDynamics::Update(double time_step, AircraftInput input)
{
    input_rk = input;
    int return_flag = 1;
    const double abserr = 1e-12;
    double relerr = 1e-6;
    // rtol=1e-6,
    //         atol=1e-12,
    double time = 0;
    // &state[0] = state_.array;
    return_flag = r8_rkf45(
        AircraftDynamics::RkFunctionDerivative,
        kNumStates,
        state_.array,
        state_dot_.array,
        &time,
        time_step,
        &relerr,
        abserr,
        return_flag);
    return return_flag;
}
double pos_x;
double pos_y;
double pos_z;
Eigen::Vector3d velocity_body;
double vel_magnitude;
Eigen::Quaterniond body_to_inertial;
Eigen::Quaterniond inertial_to_body;
Eigen::Vector3d velocity_inertial;
double omega_x;
double omega_y;
double omega_z;
Eigen::Vector3d omega_body;
double flap_l;
double flap_r;
double motor_w_l;
double motor_w_r;
Eigen::Vector3d force_gravity_body;
Eigen::Vector3d motor_thrust_l_body;
Eigen::Vector3d motor_thrust_r_body;
Eigen::Vector3d propeller_drag_l_body;
Eigen::Vector3d propeller_drag_r_body;
Eigen::Vector3d lift_body;
Eigen::Vector3d elevon_lift_reduction_l_body;
Eigen::Vector3d elevon_lift_reduction_r_body;
Eigen::Vector3d rotational_lift_wing;
Eigen::Vector3d elevon_thrust_redirection_l_body;
Eigen::Vector3d elevon_thrust_redirection_r_body;
Eigen::Vector3d rotation_elevon_lift_reduction_l_body;
Eigen::Vector3d rotation_elevon_lift_reduction_r_body;
Eigen::Vector3d total_force_body;
Eigen::Vector3d acceleration_body;
Eigen::Vector3d motor_thrust_moment_l_body;
Eigen::Vector3d motor_thrust_moment_r_body;
Eigen::Vector3d propeller_drag_moment_l_body;
Eigen::Vector3d propeller_drag_moment_r_body;
Eigen::Vector3d elevon_thrust_redirection_moment_l_body;
Eigen::Vector3d elevon_thrust_redirection_moment_r_body;
Eigen::Vector3d elevon_lift_reduction_moment_l_body;
Eigen::Vector3d elevon_lift_reduction_moment_r_body;
Eigen::Vector3d rotational_elevon_lift_reduction_moment_l_body;
Eigen::Vector3d rotational_elevon_lift_reduction_moment_r_body;
Eigen::Vector3d motor_torque_l_body;
Eigen::Vector3d motor_torque_r_body;
Eigen::Vector3d lift_moment_body;
Eigen::Vector3d propeller_drag_coeff_moment_l_body;
Eigen::Vector3d propeller_drag_coeff_moment_r_body;
Eigen::Vector3d elevon_lift_reduction_coeff_moment_body;
Eigen::Vector3d elevon_thrust_redirection_coeff_moment_l_body;
Eigen::Vector3d elevon_thrust_redirection_coeff_moment_r_body;
Eigen::Vector3d lift_moment_damping_body;
Eigen::Vector3d elevon_lift_reduction_moment_damping_body;
Eigen::Vector3d total_moment_body;
Eigen::Vector3d angular_acceleration_body;
Eigen::Matrix4d omega_dot_quat;
Eigen::Vector4d quat_dot;

void AircraftDynamics::WriteForces(std::ostream& os)
{
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ",");
    os << motor_thrust_l_body.transpose().format(fmt) << ","
       << motor_thrust_r_body.transpose().format(fmt) << ","
       << propeller_drag_l_body.transpose().format(fmt) << ","
       << propeller_drag_r_body.transpose().format(fmt) << "," << lift_body.transpose().format(fmt)
       << "," << force_gravity_body.transpose().format(fmt) << ","
       << elevon_lift_reduction_l_body.transpose().format(fmt) << ","
       << elevon_lift_reduction_r_body.transpose().format(fmt) << ","
       << rotational_lift_wing.transpose().format(fmt) << ","
       << elevon_thrust_redirection_l_body.transpose().format(fmt) << ","
       << elevon_thrust_redirection_r_body.transpose().format(fmt) << ","
       << (rotation_elevon_lift_reduction_l_body + rotation_elevon_lift_reduction_r_body)
              .transpose()
              .format(fmt);
}

void AircraftDynamics::RkFunctionDerivative(double t, double y[], double yp[])
{
    pos_x = y[0];
    pos_y = y[1];
    pos_z = y[2];

    velocity_body << y[3], y[4], y[5];
    vel_magnitude = velocity_body.norm();

    body_to_inertial.w() = y[6];
    body_to_inertial.x() = y[7];
    body_to_inertial.y() = y[8];
    body_to_inertial.z() = y[9];
    body_to_inertial.normalize();
    inertial_to_body = body_to_inertial.inverse();

    velocity_inertial = body_to_inertial * velocity_body;

    omega_x = y[10];
    omega_y = y[11];
    omega_z = y[12];
    omega_body << omega_x, omega_y, omega_z;

    flap_l = y[13];
    flap_r = y[14];
    motor_w_l = y[15];
    motor_w_r = y[16];

    if (flap_l >= aircraft_rk->max_elevon_angle && input_rk.elevon_angle_dot_left > 0)
    {
        input_rk.elevon_angle_dot_left = 0;
    }
    if (flap_r >= aircraft_rk->max_elevon_angle && input_rk.elevon_angle_dot_right > 0)
    {
        input_rk.elevon_angle_dot_right = 0;
    }
    if (flap_l <= -aircraft_rk->max_elevon_angle && input_rk.elevon_angle_dot_left < 0)
    {
        input_rk.elevon_angle_dot_left = 0;
    }
    if (flap_r <= -aircraft_rk->max_elevon_angle && input_rk.elevon_angle_dot_right < 0)
    {
        input_rk.elevon_angle_dot_right = 0;
    }

    if (motor_w_l >= 0 && input_rk.motor_omega_dot_left > 0)
    {
        input_rk.motor_omega_dot_left = 0;
    }
    if (motor_w_r >= aircraft_rk->prop_max_omega && input_rk.motor_omega_dot_right > 0)
    {
        input_rk.motor_omega_dot_right = 0;
    }
    if (motor_w_l <= -aircraft_rk->prop_max_omega && input_rk.motor_omega_dot_left < 0)
    {
        input_rk.motor_omega_dot_left = 0;
    }
    if (motor_w_r <= 0 && input_rk.motor_omega_dot_right < 0)
    {
        input_rk.motor_omega_dot_right = 0;
    }
    flap_l = std::clamp(flap_l, -aircraft_rk->max_elevon_angle, aircraft_rk->max_elevon_angle);
    flap_r = std::clamp(flap_r, -aircraft_rk->max_elevon_angle, aircraft_rk->max_elevon_angle);
    motor_w_l = std::clamp(motor_w_l, -aircraft_rk->prop_max_omega, 0.0);
    motor_w_r = std::clamp(motor_w_r, 0.0, aircraft_rk->prop_max_omega);

    // Forces
    // Gravity
    force_gravity_body = inertial_to_body * Eigen::Vector3d(0, 0, 9.81 * aircraft_rk->mass);

    // Motor thrust
    motor_thrust_l_body << aircraft_rk->prop_thrust_coeff * environment_rk->rho * motor_w_l *
                               motor_w_l,
        0.0, 0.0;
    motor_thrust_r_body << aircraft_rk->prop_thrust_coeff * environment_rk->rho * motor_w_r *
                               motor_w_r,
        0.0, 0.0;

    // Drag on the body due to propeller thrust TODO remove divide
    propeller_drag_l_body << (-0.25 * aircraft_rk->wing_surface_area /
                              aircraft_rk->propeller_disc_area) *
                                 aircraft_rk->phi_f_v * motor_thrust_l_body;
    propeller_drag_r_body << (-0.25 * aircraft_rk->wing_surface_area /
                              aircraft_rk->propeller_disc_area) *
                                 aircraft_rk->phi_f_v * motor_thrust_r_body;

    // Lift on the body as if it were non-cambered
    lift_body << -0.5 * environment_rk->rho * aircraft_rk->wing_surface_area *
                     aircraft_rk->phi_f_v * velocity_body * vel_magnitude;

    // Reduction in lift due to flap deflection
    elevon_lift_reduction_l_body << 0.25 * environment_rk->rho * aircraft_rk->wing_surface_area *
                                        aircraft_rk->phi_f_v *
                                        matrix_cross(aircraft_rk->linear_elevon_effectiveness) *
                                        flap_l * vel_magnitude * velocity_body;
    // std::cout << environment_rk->rho << std::endl
    //           << aircraft_rk->wing_surface_area << std::endl
    //           << aircraft_rk->phi_f_v << std::endl
    //           << matrix_cross(aircraft_rk->linear_elevon_effectiveness) << std::endl
    //           << flap_l << std::endl
    //           << vel_magnitude << std::endl
    //           << velocity_body << std::endl;
    // std::cout << elevon_lift_reduction_l_body << std::endl;
    // std::cin.get();
    elevon_lift_reduction_r_body << 0.25 * environment_rk->rho * aircraft_rk->wing_surface_area *
                                        aircraft_rk->phi_f_v *
                                        matrix_cross(aircraft_rk->linear_elevon_effectiveness) *
                                        flap_r * vel_magnitude * velocity_body;

    // Lift on the body due to the body rotating in the air
    rotational_lift_wing << -0.5 * environment_rk->rho * aircraft_rk->wing_surface_area *
                                vel_magnitude * aircraft_rk->phi_m_v * aircraft_rk->B * omega_body;

    // Force caused by elevon thrust redirection
    elevon_thrust_redirection_l_body
        << 0.25 * aircraft_rk->wing_surface_area / aircraft_rk->propeller_disc_area *
               aircraft_rk->phi_f_v * matrix_cross(aircraft_rk->linear_elevon_effectiveness) *
               flap_l * motor_thrust_l_body;
    elevon_thrust_redirection_r_body
        << 0.25 * aircraft_rk->wing_surface_area / aircraft_rk->propeller_disc_area *
               aircraft_rk->phi_f_v * matrix_cross(aircraft_rk->linear_elevon_effectiveness) *
               flap_r * motor_thrust_r_body;

    // Reduction in rotational lift due to elevons
    rotation_elevon_lift_reduction_l_body
        << 0.25 * environment_rk->rho * aircraft_rk->wing_surface_area * aircraft_rk->phi_m_v *
               matrix_cross(aircraft_rk->linear_elevon_effectiveness) * flap_l * vel_magnitude *
               aircraft_rk->B * omega_body;
    rotation_elevon_lift_reduction_r_body
        << 0.25 * environment_rk->rho * aircraft_rk->wing_surface_area * aircraft_rk->phi_m_v *
               matrix_cross(aircraft_rk->linear_elevon_effectiveness) * flap_r * vel_magnitude *
               aircraft_rk->B * omega_body;

    // Total Force and Acceleration
    total_force_body = motor_thrust_l_body + motor_thrust_r_body + propeller_drag_l_body +
                       propeller_drag_r_body + lift_body + force_gravity_body +
                       elevon_lift_reduction_l_body + elevon_lift_reduction_r_body +
                       rotational_lift_wing + elevon_thrust_redirection_l_body +
                       elevon_thrust_redirection_r_body + rotation_elevon_lift_reduction_l_body +
                       rotation_elevon_lift_reduction_r_body;

    acceleration_body =
        total_force_body / aircraft_rk->mass - matrix_cross(omega_body) * velocity_body;

    /******** Moments ********/
    // Moments from forces
    motor_thrust_moment_l_body =
        matrix_cross(aircraft_rk->propeller_moment_arm_left) * motor_thrust_l_body;
    motor_thrust_moment_r_body =
        matrix_cross(aircraft_rk->propeller_moment_arm_right) * motor_thrust_r_body;

    propeller_drag_moment_l_body =
        matrix_cross(aircraft_rk->aerodynamic_moment_arm_left) * propeller_drag_l_body;
    propeller_drag_moment_r_body =
        matrix_cross(aircraft_rk->aerodynamic_moment_arm_right) * propeller_drag_r_body;

    elevon_thrust_redirection_moment_l_body =
        matrix_cross(aircraft_rk->aerodynamic_moment_arm_left) * elevon_thrust_redirection_l_body;
    elevon_thrust_redirection_moment_r_body =
        matrix_cross(aircraft_rk->aerodynamic_moment_arm_right) * elevon_thrust_redirection_r_body;

    elevon_lift_reduction_moment_l_body =
        matrix_cross(aircraft_rk->aerodynamic_moment_arm_left) * elevon_lift_reduction_l_body;
    // std::cout << matrix_cross(aircraft_rk->aerodynamic_moment_arm_left) << std::endl
    //           << elevon_lift_reduction_l_body << std::endl;
    // std::cin.get();
    elevon_lift_reduction_moment_r_body =
        matrix_cross(aircraft_rk->aerodynamic_moment_arm_right) * elevon_lift_reduction_r_body;

    rotational_elevon_lift_reduction_moment_l_body =
        matrix_cross(aircraft_rk->aerodynamic_moment_arm_left) *
        rotation_elevon_lift_reduction_l_body;
    rotational_elevon_lift_reduction_moment_r_body =
        matrix_cross(aircraft_rk->aerodynamic_moment_arm_right) *
        rotation_elevon_lift_reduction_r_body;

    // Moments from Motor Torque
    motor_torque_l_body << -signum(motor_w_l) * aircraft_rk->prop_moment_coeff * motor_w_l *
                               motor_w_l,
        0, 0;
    motor_torque_r_body << -signum(motor_w_r) * aircraft_rk->prop_moment_coeff * motor_w_r *
                               motor_w_r,
        0, 0;

    // Moments from moment coefficients

    // phi_mv moments
    lift_moment_body = 0.5 * environment_rk->rho * aircraft_rk->wing_surface_area * aircraft_rk->B *
                       aircraft_rk->phi_m_v * vel_magnitude * velocity_body;
    propeller_drag_coeff_moment_l_body = -0.25 * aircraft_rk->wing_surface_area *
                                         aircraft_rk->propeller_disc_area * aircraft_rk->B *
                                         aircraft_rk->phi_m_v * motor_thrust_l_body;
    propeller_drag_coeff_moment_r_body = -0.25 * aircraft_rk->wing_surface_area *
                                         aircraft_rk->propeller_disc_area * aircraft_rk->B *
                                         aircraft_rk->phi_m_v * motor_thrust_r_body;
    elevon_lift_reduction_coeff_moment_body =
        0.25 * environment_rk->rho * aircraft_rk->wing_surface_area * aircraft_rk->B *
        aircraft_rk->phi_m_v * matrix_cross(aircraft_rk->rotational_elevon_effectiveness) *
        (flap_l + flap_r) * vel_magnitude * velocity_body;

    elevon_thrust_redirection_coeff_moment_l_body =
        0.25 * aircraft_rk->wing_surface_area / aircraft_rk->propeller_disc_area * aircraft_rk->B *
        aircraft_rk->phi_m_v * matrix_cross(aircraft_rk->rotational_elevon_effectiveness) * flap_l *
        motor_thrust_l_body;
    elevon_thrust_redirection_coeff_moment_r_body =
        0.25 * aircraft_rk->wing_surface_area / aircraft_rk->propeller_disc_area * aircraft_rk->B *
        aircraft_rk->phi_m_v * matrix_cross(aircraft_rk->rotational_elevon_effectiveness) * flap_r *
        motor_thrust_r_body;
    // phi_m_omega moments
    lift_moment_damping_body = -0.5 * environment_rk->rho * aircraft_rk->wing_surface_area *
                               aircraft_rk->B * aircraft_rk->phi_m_omega * vel_magnitude *
                               aircraft_rk->B * omega_body;
    elevon_lift_reduction_moment_damping_body =
        0.25 * environment_rk->rho * aircraft_rk->wing_surface_area * aircraft_rk->B *
        aircraft_rk->phi_m_omega * matrix_cross(aircraft_rk->rotational_elevon_effectiveness) *
        (flap_l + flap_r) * vel_magnitude * aircraft_rk->B * omega_body;

    // Total moment and angular acceleration
    total_moment_body =
        motor_thrust_moment_l_body + motor_thrust_moment_r_body + propeller_drag_moment_l_body +
        propeller_drag_moment_r_body + elevon_thrust_redirection_moment_l_body +
        elevon_thrust_redirection_moment_r_body + elevon_lift_reduction_moment_l_body +
        elevon_lift_reduction_moment_r_body + rotational_elevon_lift_reduction_moment_l_body +
        rotational_elevon_lift_reduction_moment_r_body + motor_torque_l_body + motor_torque_r_body +
        lift_moment_body + propeller_drag_coeff_moment_l_body + propeller_drag_coeff_moment_r_body +
        elevon_lift_reduction_coeff_moment_body + elevon_thrust_redirection_coeff_moment_l_body +
        elevon_thrust_redirection_coeff_moment_r_body + lift_moment_damping_body +
        elevon_lift_reduction_moment_damping_body;

    angular_acceleration_body =
        aircraft_rk->moment_of_inertia_inv * total_moment_body -
        aircraft_rk->moment_of_inertia_inv *
            (matrix_cross(omega_body) * aircraft_rk->moment_of_inertia * omega_body);

    // quat_dot calc
    omega_dot_quat << 0.0, -omega_x, -omega_y, -omega_z, omega_x, 0.0, omega_z, -omega_y, omega_y,
        -omega_z, 0.0, omega_x, omega_z, omega_y, -omega_x, 0.0;
    quat_dot = quat_dot = 0.5 * omega_dot_quat * Eigen::Vector4d(y[6], y[7], y[8], y[9]);

    // update dynamics
    yp[0] = velocity_inertial[0];
    yp[1] = velocity_inertial[1];
    yp[2] = velocity_inertial[2];
    yp[3] = acceleration_body[0];
    yp[4] = acceleration_body[1];
    yp[5] = acceleration_body[2];
    yp[6] = quat_dot[0];
    yp[7] = quat_dot[1];
    yp[8] = quat_dot[2];
    yp[9] = quat_dot[3];
    yp[10] = angular_acceleration_body[0];
    yp[11] = angular_acceleration_body[1];
    yp[12] = angular_acceleration_body[2];
    yp[13] = input_rk.elevon_angle_dot_left;
    yp[14] = input_rk.elevon_angle_dot_right;
    yp[15] = input_rk.motor_omega_dot_left;
    yp[16] = input_rk.motor_omega_dot_right;
    // for (int ii = 0; ii < kNumStates; ++ii)
    // {
    //     std::cout << yp[ii] << "\t";
    // }
    // std::cout << std::endl;
}
