#pragma once
#include "aircraft_controller_params.hpp"
#include "aircraft_dynamics.hpp"
class TrackingController
{
   public:
    struct AircraftControllerInput
    {
        double elevon_angle_left;
        double elevon_angle_right;
        double motor_omega_left;
        double motor_omega_right;
        friend std::ostream& operator<<(
            std::ostream& os,
            const AircraftControllerInput& aircraft_input)
        {
            os << aircraft_input.elevon_angle_left << "," << aircraft_input.elevon_angle_right
               << "," << aircraft_input.motor_omega_left << "," << aircraft_input.motor_omega_right;
            return os;
        }
    };

    struct AircraftControllerDesired
    {
        Eigen::Vector3d pos_des;
        Eigen::Vector3d vel_des;
        Eigen::Vector3d acc_des;
        Eigen::Vector3d jerk_des;
        double yaw_des;
        double yaw_rate_des;
    };
    TrackingController(
        AircraftControllerParameters params,
        std::string output_file = "control.csv");
    ~TrackingController();
    AircraftDynamics::AircraftInput Update(
        Eigen::Vector3d pos_des,
        Eigen::Vector3d vel_des,
        Eigen::Vector3d acc_des,
        Eigen::Vector3d jerk_des,
        double yaw_des,
        double yaw_rate_des,
        const AircraftDynamics::AircraftState current_state,
        const AircraftDynamics::AircraftInput prev_input,
        Eigen::Vector3d accelerometer_measurement,
        Eigen::Vector3d omega_dot);
    AircraftDynamics::AircraftInput Update(
        AircraftControllerDesired desired,
        const AircraftDynamics::AircraftState current_state,
        const AircraftDynamics::AircraftInput prev_input,
        Eigen::Vector3d accelerometer_measurement,
        Eigen::Vector3d omega_dot);
    AircraftDynamics::AircraftInput Update(
        const AircraftDynamics::AircraftState current_state,
        const AircraftDynamics::AircraftInput prev_input,
        Eigen::Vector3d accelerometer_measurement,
        Eigen::Vector3d omega_dot);
    void UpdateDesired(
        Eigen::Vector3d pos_des,
        Eigen::Vector3d vel_des,
        Eigen::Vector3d acc_des,
        Eigen::Vector3d jerk_des,
        double yaw_des,
        double yaw_rate_des);

    void UpdateDesired(AircraftControllerDesired desired);

    void UpdateEstimates();
    Eigen::Vector3d ControlPositionVelocity();
    void ControlLinearAcceleration(Eigen::Vector3d acc_cmd);
    std::pair<Eigen::Quaterniond, double> ForceYawTransform();
    Eigen::Vector3d DiffFlatnessJerkYawRateTransform();
    Eigen::Vector3d ControlAttitudeAngularRate(
        Eigen::Quaterniond quat_des,
        Eigen::Vector3d omega_des);
    Eigen::Vector3d ControlAngularAcceleration(Eigen::Vector3d omega_dot_des);
    AircraftControllerInput ThrustMomentTransform(double thrust_des, Eigen::Vector3d moment_des);

   private:
    double FlapPositionController(double des_angle, double actual_angle);
    double MotorVelocityController(double des_velocity, double actual_velocity);
    std::ofstream control_log_;
    bool log_controls_;
    double time_ = 0;
    double dt_;
    AircraftControllerParameters params_;
    // update input
    Eigen::Vector3d pos_des_;
    Eigen::Vector3d vel_des_;
    Eigen::Vector3d acc_des_;
    Eigen::Vector3d jerk_des_;
    double yaw_des_;
    double yaw_rate_des_;
    AircraftDynamics::AircraftInput current_control_;
    AircraftDynamics::AircraftState current_state_;
    Eigen::Vector3d acceleration_measurement_;
    Eigen::Vector3d angular_acceleration_measurement_;

    // States
    // linear
    Eigen::Vector3d velocity_body_;
    Eigen::Vector3d velocity_inertial_;
    double velocity_magnitude_;

    Eigen::Vector3d acceleration_tilda_lpf_;
    // rotational
    Eigen::Quaterniond body_to_inertial_;
    Eigen::Quaterniond inertial_to_body_;

    // filtered data
    Eigen::Vector3d acceleration_filt_;
    Eigen::Vector3d omega_body_filt_;
    Eigen::Vector3d omega_body_filt_prev_;
    Eigen::Vector3d omega_dot_body_filt_;
    double flap_l_filt_;
    double flap_r_filt_;
    // Forces and Moments
    // Thrust Forces
    // Flap Forces
    // Eigen::Vector3d force_flaps_body_filt_;
    Eigen::Vector3d force_inertial_lpf_;
    // Wing
    // Moments
    Eigen::Vector3d moment_body_filt_;

    // things to pass around for calculations
    double phi_ = 0;
    double theta_;
    double beta_x_;
    double beta_z_;
    double sigma_x_;
    double sigma_z_;
    double eta_;
    Eigen::Vector3d force_command_;
    Eigen::Vector3d vel_phi_;
    Eigen::Quaterniond inertial_to_phi_rotation_;

    // Commands
    // Eigen::Vector3d moment_commanded_;
    // Eigen::Vector3d omega_dot_commanded_;
    // Eigen::Vector3d omega_des_;
    // Eigen::Quaterniond quat_des_;
    // Eigen::Vector3d thrust_des_;
};