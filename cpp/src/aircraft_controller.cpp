#include "aircraft_controller.hpp"

TrackingController::TrackingController(AircraftControllerParameters params, std::string output_file)
    : params_(params), omega_body_filt_prev_(Eigen::Vector3d::Zero()), log_controls_(false)
{
    dt_ = 1.0 / params_.controller_frequency;
    control_log_.open(output_file);
    if (control_log_.is_open())
    {
        log_controls_ = true;
        control_log_
            << "t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z,omega_x,omega_y,"
               "omega_z,elevon_left,elevon_right,motor_omega_left,motor_omega_right"
            << std::endl;
    } else
    {
        log_controls_ = false;
        std::cerr << "Error: Unable to open file output file " << output_file << std::endl;
    }
}
TrackingController::~TrackingController()
{
    if (log_controls_)
    {
        control_log_.close();
    }
}

AircraftDynamics::AircraftInput TrackingController::Update(
    Eigen::Vector3d pos_des,
    Eigen::Vector3d vel_des,
    Eigen::Vector3d acc_des,
    Eigen::Vector3d jerk_des,
    double yaw_des,
    double yaw_rate_des,
    const AircraftDynamics::AircraftState current_state,
    const AircraftDynamics::AircraftInput prev_input,
    Eigen::Vector3d accelerometer_measurement,
    Eigen::Vector3d omega_dot)
{
    UpdateDesired(pos_des, vel_des, acc_des, jerk_des, yaw_des, yaw_rate_des);
    return Update(current_state, prev_input, accelerometer_measurement, omega_dot);
}
AircraftDynamics::AircraftInput TrackingController::Update(
    AircraftControllerDesired desired,
    const AircraftDynamics::AircraftState current_state,
    const AircraftDynamics::AircraftInput prev_input,
    Eigen::Vector3d accelerometer_measurement,
    Eigen::Vector3d omega_dot)
{
    UpdateDesired(desired);
    return Update(current_state, prev_input, accelerometer_measurement, omega_dot);
}
AircraftDynamics::AircraftInput TrackingController::Update(
    const AircraftDynamics::AircraftState current_state,
    const AircraftDynamics::AircraftInput prev_input,
    Eigen::Vector3d accelerometer_measurement,
    Eigen::Vector3d omega_dot)
{
    AircraftDynamics::AircraftInput input;
    time_ += dt_;
    memcpy(current_state_.array, current_state.array, sizeof(current_state.array));
    acceleration_measurement_ = accelerometer_measurement;
    angular_acceleration_measurement_ = omega_dot;
    UpdateEstimates();
    Eigen::Vector3d acceleration_command = ControlPositionVelocity();
    ControlLinearAcceleration(acceleration_command);
    std::pair<Eigen::Quaterniond, double> force_yaw_out = ForceYawTransform();
    Eigen::Vector3d omega_des = DiffFlatnessJerkYawRateTransform();
    Eigen::Vector3d omega_dot_des = ControlAttitudeAngularRate(force_yaw_out.first, omega_des);
    Eigen::Quaterniond q_des = force_yaw_out.first;
    double thrust_des = force_yaw_out.second;
    // q_des = Eigen::AngleAxisd(0.0 * time_, Eigen::Vector3d::UnitZ()) *
    //         Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
    //         Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY());
    // Eigen::Vector3d omega_dot_des = ControlAttitudeAngularRate(q_des, Eigen::Vector3d::Zero());
    Eigen::Vector3d moment_command = ControlAngularAcceleration(omega_dot_des);
    // std::cout << "Moment Command:" << std::endl << moment_command << std::endl << std::endl;
    TrackingController::AircraftControllerInput controller_output =
        ThrustMomentTransform(thrust_des, moment_command);
    // std::cout << "Control: " << std::endl << controller_output << std::endl << std::endl;
    // std::cin.get();
    if (log_controls_)
    {
        // t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z,omega_x,omega_y,omega_z,elevon_left,elevon_right,motor_omega_left,motor_omega_right
        control_log_ << time_ << "," << pos_des_.x() << "," << pos_des_.y() << "," << pos_des_.z()
                     << "," << vel_des_.x() << "," << vel_des_.y() << "," << vel_des_.z() << ","
                     << q_des.w() << "," << q_des.x() << "," << q_des.y() << "," << q_des.z() << ","
                     << omega_des.x() << "," << omega_des.y() << "," << omega_des.z() << ","
                     << controller_output.elevon_angle_left << ","
                     << controller_output.elevon_angle_right << ","
                     << controller_output.motor_omega_left << ","
                     << controller_output.motor_omega_right << std::endl;
    }
    input.elevon_angle_dot_left = FlapPositionController(
        controller_output.elevon_angle_left, current_state_.elevon_angle_left);
    input.elevon_angle_dot_right = FlapPositionController(
        controller_output.elevon_angle_right, current_state_.elevon_angle_right);
    input.motor_omega_dot_left = MotorVelocityController(
        controller_output.motor_omega_left, current_state_.motor_omega_left);
    input.motor_omega_dot_right = MotorVelocityController(
        controller_output.motor_omega_right, current_state_.motor_omega_right);
    return input;
}

void TrackingController::UpdateEstimates()
{
    body_to_inertial_ = Eigen::Quaterniond(
        current_state_.quat_w, current_state_.quat_x, current_state_.quat_y, current_state_.quat_z);
    inertial_to_body_ = body_to_inertial_.conjugate();
    omega_body_filt_ = Eigen::Vector3d(
        current_state_.omega_x,
        current_state_.omega_y,
        current_state_.omega_z);  // TODO: filter this
    omega_dot_body_filt_ =
        (omega_body_filt_ - omega_body_filt_prev_) * params_.controller_frequency;
    omega_body_filt_prev_ = omega_body_filt_;
    velocity_body_ = Eigen::Vector3d(
        current_state_.velocity_x, current_state_.velocity_y, current_state_.velocity_z);
    velocity_inertial_ = body_to_inertial_ * velocity_body_;
    velocity_magnitude_ = velocity_body_.norm();

    Eigen::Vector3d acceleration_data_minus_gravity = body_to_inertial_ * acceleration_measurement_;
    acceleration_filt_ = acceleration_data_minus_gravity;
    double motor_w_l_filt = current_state_.motor_omega_left;
    double motor_w_r_filt = current_state_.motor_omega_right;
    flap_l_filt_ = current_state_.elevon_angle_left;
    flap_r_filt_ = current_state_.elevon_angle_right;
    // calculate force estimates

    // thrust
    double motor_thrust_l_body_filt =
        params_.model_params.prop_thrust_coeff * pow(motor_w_l_filt, 2);
    double motor_thrust_r_body_filt =
        params_.model_params.prop_thrust_coeff * pow(motor_w_r_filt, 2);

    double force_thrust_l_body_filt =
        (1 - params_.model_params.wing_drag_coeff_thrust) * motor_thrust_l_body_filt;
    double force_thrust_r_body_filt =
        (1 - params_.model_params.wing_drag_coeff_thrust) * motor_thrust_r_body_filt;
    Eigen::Vector3d force_thrust_body_filt(
        force_thrust_l_body_filt + force_thrust_r_body_filt, 0, 0);

    // flaps force
    double force_flaps_l_body_lpf_hpf =
        -flap_l_filt_ * (params_.model_params.elevon_lift_coeff_thrust * motor_thrust_l_body_filt +
                         params_.model_params.elevon_lift_coeff_airspeed * velocity_magnitude_ *
                             current_state_.velocity_x);
    double force_flaps_r_body_lpf_hpf =
        -flap_r_filt_ * (params_.model_params.elevon_lift_coeff_thrust * motor_thrust_r_body_filt +
                         params_.model_params.elevon_lift_coeff_airspeed * velocity_magnitude_ *
                             current_state_.velocity_x);
    Eigen::Vector3d force_flaps_body_lpf_hpf = Eigen::Vector3d::Zero();
    force_flaps_body_lpf_hpf(2) = force_flaps_l_body_lpf_hpf + force_flaps_r_body_lpf_hpf;

    // wing force
    Eigen::Vector3d force_wing_body_filt(
        -params_.model_params.wing_drag_coeff * current_state_.velocity_x,
        0,
        -params_.model_params.wing_drag_coeff * current_state_.velocity_z);
    force_wing_body_filt *= velocity_magnitude_;

    Eigen::Vector3d force_body_filt =
        force_thrust_body_filt + force_flaps_body_lpf_hpf + force_wing_body_filt;

    // caclulate moments
    // thrust
    Eigen::Vector3d thrust_moment_body_filt(
        0,
        params_.model_params.pitch_coeff_thrust *
            (motor_thrust_l_body_filt + motor_thrust_r_body_filt),
        params_.model_params.thrust_l_y * (force_thrust_l_body_filt - force_thrust_r_body_filt));

    // motor torque
    double motor_torque_l_body = -params_.model_params.prop_moment_coeff * pow(motor_w_l_filt, 2);
    double motor_torque_r_body = params_.model_params.prop_moment_coeff * pow(motor_w_r_filt, 2);

    Eigen::Vector3d motor_torque_moment_body_filt(motor_torque_l_body + motor_torque_r_body, 0, 0);

    // flap moment
    Eigen::Vector3d flap_moment_body(
        params_.model_params.flap_length_y *
            (force_flaps_r_body_lpf_hpf - force_flaps_l_body_lpf_hpf),
        params_.model_params.flap_length_x *
            (force_flaps_r_body_lpf_hpf + force_flaps_l_body_lpf_hpf),
        0);
    // wing_moment = matrix_cross(np.array([-0.015, 0, 0])) @ force_wing_body_filt

    moment_body_filt_ = thrust_moment_body_filt + motor_torque_moment_body_filt +
                        flap_moment_body;  // todo look at wing moment

    force_inertial_lpf_ = body_to_inertial_ * force_body_filt;
    acceleration_tilda_lpf_ = acceleration_filt_ - body_to_inertial_ * force_flaps_body_lpf_hpf /
                                                       params_.model_params.mass;
}

Eigen::Vector3d TrackingController::ControlPositionVelocity()
{
    Eigen::Vector3d pos_correction =
        params_.pos_gain * inertial_to_body_ *
        (pos_des_ -
         Eigen::Vector3d(
             current_state_.position_x, current_state_.position_y, current_state_.position_z));
    Eigen::Vector3d vel_correction =
        params_.vel_gain * inertial_to_body_ * (vel_des_ - velocity_inertial_);
    Eigen::Vector3d acc_correction =
        params_.acc_gain * inertial_to_body_ * (acc_des_ - acceleration_tilda_lpf_);

    Eigen::Vector3d acceleration_command =
        body_to_inertial_ * (pos_correction + vel_correction + acc_correction) + acc_des_;

    return acceleration_command;
}

void TrackingController::ControlLinearAcceleration(Eigen::Vector3d acc_cmd)
{
    force_command_ =
        (acc_cmd - acceleration_tilda_lpf_) * params_.model_params.mass + force_inertial_lpf_;
}

std::pair<Eigen::Quaterniond, double> TrackingController::ForceYawTransform()
{
    Eigen::AngleAxisd inertial_to_yaw_rotation(-yaw_des_, Eigen::Vector3d::UnitZ());
    beta_x_ = (inertial_to_yaw_rotation * force_command_)(1);
    beta_z_ = force_command_(2);
    phi_ = -atan2(beta_x_, beta_z_);
    Eigen::AngleAxisd phi_rotation(-phi_, Eigen::Vector3d::UnitX());
    inertial_to_phi_rotation_ = phi_rotation * inertial_to_yaw_rotation;
    Eigen::Vector3d b_y = body_to_inertial_ * Eigen::Vector3d::UnitY();
    Eigen::Vector3d phi_y = inertial_to_phi_rotation_.conjugate() * Eigen::Vector3d::UnitY();
    double dot_y = b_y.dot(phi_y);
    if (dot_y <= 0.0)
    {
        phi_ += M_PI;
        phi_rotation = Eigen::AngleAxisd(-phi_, Eigen::Vector3d::UnitX());
        inertial_to_phi_rotation_ = phi_rotation * inertial_to_yaw_rotation;
        if (phi_ > M_PI)
        {
            phi_ = -(M_PI * 2 - phi_);
        }
    }

    Eigen::Vector3d force_phi = inertial_to_phi_rotation_ * force_command_;
    Eigen::Vector3d vel_phi = inertial_to_phi_rotation_ * velocity_inertial_;
    double delta = flap_l_filt_ + flap_r_filt_;
    eta_ = (-params_.model_params.elevon_lift_coeff_thrust * delta / 2) /
           (1 - params_.model_params.wing_drag_coeff_thrust);
    sigma_x_ =
        eta_ * (force_phi[0] +
                params_.model_params.wing_drag_coeff * velocity_magnitude_ * vel_phi[0]) -
        params_.model_params.elevon_lift_coeff_airspeed * delta * velocity_magnitude_ * vel_phi[0] -
        params_.model_params.wing_lift_coeff * velocity_magnitude_ * vel_phi[2] - force_phi[2];
    sigma_z_ =
        eta_ * (force_phi[2] +
                params_.model_params.wing_drag_coeff * velocity_magnitude_ * vel_phi[2]) -
        params_.model_params.elevon_lift_coeff_airspeed * delta * velocity_magnitude_ * vel_phi[2] +
        params_.model_params.wing_lift_coeff * velocity_magnitude_ * vel_phi[0] + force_phi[0];

    theta_ = atan2(sigma_x_, sigma_z_);

    double c_theta = cos(theta_);
    double s_theta = sin(theta_);

    double thrust = 1 / (1 - params_.model_params.wing_drag_coeff_thrust) *
                    (c_theta * force_phi[0] - s_theta * force_phi[2] +
                     params_.model_params.wing_drag_coeff * velocity_magnitude_ *
                         (c_theta * vel_phi[0] - s_theta * vel_phi[2]));

    if (thrust < 0.0)
    {
        theta_ += M_PI;
        c_theta = cos(theta_);
        s_theta = sin(theta_);

        thrust = 1 / (1 - params_.model_params.wing_drag_coeff_thrust) *
                 (c_theta * force_phi[0] - s_theta * force_phi[2] +
                  params_.model_params.wing_drag_coeff * velocity_magnitude_ *
                      (c_theta * vel_phi[0] - s_theta * vel_phi[2]));
    }

    Eigen::Quaterniond q_des = Eigen::AngleAxisd(yaw_des_, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(phi_, Eigen::Vector3d::UnitX()) *
                               Eigen::AngleAxisd(theta_, Eigen::Vector3d::UnitY());
    return std::pair<Eigen::Quaterniond, double>(q_des, thrust);
}

Eigen::Vector3d TrackingController::DiffFlatnessJerkYawRateTransform()
{
    Eigen::Vector3d force_dot = params_.model_params.mass * jerk_des_;
    double cyaw = cos(yaw_des_);
    double syaw = sin(yaw_des_);
    double cphi = cos(phi_);
    double sphi = sin(phi_);
    double beta_x_dot = -cyaw * yaw_rate_des_ * force_dot(0) - syaw * force_dot(0) -
                        syaw * yaw_rate_des_ * force_dot(1) + cyaw * force_dot(1);
    double beta_z_dot = force_dot(2);

    double phi_dot_des =
        -(beta_x_dot * beta_z_ - beta_x_ * beta_z_dot) / (pow(beta_x_, 2) + pow(beta_z_, 2));

    double vel_des_magnitude = vel_des_.norm();
    double vel_des_dot_magnitude;
    if (vel_des_magnitude != 0.0)
    {
        vel_des_dot_magnitude = vel_des_.transpose() * acc_des_;
        vel_des_dot_magnitude /= vel_des_magnitude;
    } else
    {
        vel_des_dot_magnitude = 0.0;
    }

    Eigen::Matrix3d intertial_to_phi_dot;
    intertial_to_phi_dot << -yaw_rate_des_ * syaw, yaw_rate_des_ * cyaw, 0,
        phi_dot_des * sphi * syaw - yaw_rate_des_ * cphi * cyaw,
        -phi_dot_des * sphi * cyaw - yaw_rate_des_ * syaw * cphi, phi_dot_des * cphi,
        phi_dot_des * cphi * syaw + yaw_rate_des_ * sphi * cyaw,
        -phi_dot_des * cphi * cyaw + yaw_rate_des_ * syaw * sphi, -phi_dot_des * sphi;

    Eigen::Vector3d vel_dot_phi =
        intertial_to_phi_dot * vel_des_ + inertial_to_phi_rotation_ * acc_des_;
    Eigen::Vector3d force_dot_phi =
        intertial_to_phi_dot * force_command_ + inertial_to_phi_rotation_ * force_dot;

    double tau_x = vel_des_dot_magnitude * vel_phi_[0] + vel_des_magnitude * vel_dot_phi[0];
    double tau_z = vel_des_dot_magnitude * vel_phi_[2] + vel_des_magnitude * vel_dot_phi[2];
    double delta = flap_l_filt_ + flap_r_filt_;

    double sigma_x_dot = eta_ * (force_dot_phi[0] + params_.model_params.wing_drag_coeff * tau_x) -
                         params_.model_params.elevon_lift_coeff_airspeed * delta * tau_x -
                         params_.model_params.wing_lift_coeff * tau_z - force_dot_phi[2];
    double sigma_z_dot = eta_ * (force_dot_phi[2] + params_.model_params.wing_drag_coeff * tau_z) -
                         params_.model_params.elevon_lift_coeff_airspeed * delta * tau_z -
                         params_.model_params.wing_lift_coeff * tau_x - force_dot_phi[0];
    double theta_dot =
        (sigma_x_dot * sigma_z_ - sigma_x_ * sigma_z_dot) / (pow(sigma_x_, 2) + pow(sigma_z_, 2));
    Eigen::Vector3d omega_theta(0, -theta_dot, 0);

    Eigen::AngleAxisd rot_about_theta(-theta_, Eigen::Vector3d::UnitY());
    Eigen::Vector3d omega_phi = rot_about_theta * Eigen::Vector3d(-phi_dot_des, 0, 0);
    Eigen::Vector3d omega_yaw = rot_about_theta * Eigen::Vector3d(0, 0, yaw_rate_des_);

    Eigen::Vector3d omega_des = omega_theta + omega_phi + omega_yaw;
    return omega_des;
}
Eigen::Vector3d TrackingController::ControlAttitudeAngularRate(
    Eigen::Quaterniond quat_des,
    Eigen::Vector3d omega_des)
{
    Eigen::Quaterniond current_quat(
        current_state_.quat_w, current_state_.quat_x, current_state_.quat_y, current_state_.quat_z);
    Eigen::Quaterniond error_quat = current_quat.conjugate() * quat_des;
    Eigen::Vector3d angle_error_vector;
    if (error_quat.w() >= 1.0)
    {
        angle_error_vector = Eigen::Vector3d::Zero();
    } else
    {
        double bottom = sqrt(1 - pow(error_quat.w(), 2));
        angle_error_vector = 2 * acos(error_quat.w()) / bottom *
                             Eigen::Vector3d(error_quat.x(), error_quat.y(), error_quat.z());
    }
    Eigen::Vector3d omega_dot_commanded = params_.quat_gain * angle_error_vector +
                                          params_.omega_gain * (omega_des - omega_body_filt_);
    // std::cout << omega_dot_commanded[0] << "\t" << omega_dot_commanded[1] << "\t"
    //           << omega_dot_commanded[2];
    return omega_dot_commanded;
}
Eigen::Vector3d TrackingController::ControlAngularAcceleration(Eigen::Vector3d omega_dot_des)
{
    Eigen::Vector3d moment_command =
        params_.model_params.moment_of_inertia * (omega_dot_des - omega_dot_body_filt_) +
        moment_body_filt_;
    return moment_command;
}

TrackingController::AircraftControllerInput TrackingController::ThrustMomentTransform(
    double thrust_des,
    Eigen::Vector3d moment_des)
{
    double delta_thrust = moment_des[2] / (params_.model_params.thrust_l_y *
                                           (1 - params_.model_params.wing_drag_coeff_thrust));
    double thrust_l_body = (thrust_des + delta_thrust) * 0.5;
    double thrust_r_body = (thrust_des - delta_thrust) * 0.5;
    thrust_l_body = std::max(thrust_l_body, 0.0);  // clamp to above 0
    thrust_r_body = std::max(thrust_r_body, 0.0);

    double motor_w_l = -sqrt(thrust_l_body / (params_.model_params.prop_thrust_coeff));
    double motor_w_r = sqrt(thrust_r_body / (params_.model_params.prop_thrust_coeff));

    // This is the moment due to thrust. Now we must match the remaining moment with flaps
    Eigen::Vector3d thrust_moment_body_des(
        0.0,
        params_.model_params.pitch_coeff_thrust * (thrust_l_body + thrust_r_body),
        params_.model_params.thrust_l_y * (thrust_l_body - thrust_r_body));

    double motor_torque_l_body = -params_.model_params.prop_moment_coeff * pow(motor_w_l, 2);
    double motor_torque_r_body = params_.model_params.prop_moment_coeff * pow(motor_w_r, 2);

    Eigen::Vector3d motor_torque_moment_body_des(
        motor_torque_l_body + motor_torque_r_body, 0.0, 0.0);
    Eigen::Vector3d moment_flaps_des(
        moment_des - thrust_moment_body_des - motor_torque_moment_body_des);

    double v_left =
        -params_.model_params.elevon_lift_coeff_thrust * thrust_l_body -
        params_.model_params.elevon_lift_coeff_airspeed * velocity_magnitude_ * velocity_body_[0];
    double v_right =
        -params_.model_params.elevon_lift_coeff_thrust * thrust_r_body -
        params_.model_params.elevon_lift_coeff_airspeed * velocity_magnitude_ * velocity_body_[0];

    Eigen::Matrix2d flap_pre_1;
    flap_pre_1 << -params_.model_params.flap_length_y * v_left,
        params_.model_params.flap_length_y * v_right, params_.model_params.flap_length_x * v_left,
        params_.model_params.flap_length_x * v_right;
    Eigen::Vector2d flap_pre_2(moment_flaps_des[0], moment_flaps_des[1]);
    Eigen::Vector2d flaps = flap_pre_1.inverse() * flap_pre_2;

    double flap_l = flaps[0];
    double flap_r = flaps[1];
    flap_l = std::clamp(
        flap_l, -params_.model_params.max_elevon_angle, params_.model_params.max_elevon_angle);
    flap_r = std::clamp(
        flap_r, -params_.model_params.max_elevon_angle, params_.model_params.max_elevon_angle);
    motor_w_l = std::clamp(motor_w_l, -params_.model_params.prop_max_omega, 0.0);
    motor_w_r = std::clamp(motor_w_r, 0.0, params_.model_params.prop_max_omega);
    TrackingController::AircraftControllerInput final_outputs = {
        .elevon_angle_left = flap_l,
        .elevon_angle_right = flap_r,
        .motor_omega_left = motor_w_l,
        .motor_omega_right = motor_w_r};
    // std::cout << flap_l << "\t" << flap_r << "\t" << motor_w_l << "\t" << motor_w_r;

    return final_outputs;
}

double TrackingController::FlapPositionController(double des_angle, double actual_angle)
{
    double max_vel = 40000;
    double kP = 20;
    double error = des_angle - actual_angle;
    double flap_dot = std::clamp(error * kP, -max_vel * dt_, max_vel * dt_);
    return flap_dot;
}
double TrackingController::MotorVelocityController(double des_velocity, double actual_velocity)
{
    double max_acc = 16000000;
    double kP = 20;
    double error = des_velocity - actual_velocity;
    double motor_dot = std::clamp(error * kP, -max_acc * dt_, max_acc * dt_);
    return motor_dot;
}

void TrackingController::UpdateDesired(
    Eigen::Vector3d pos_des,
    Eigen::Vector3d vel_des,
    Eigen::Vector3d acc_des,
    Eigen::Vector3d jerk_des,
    double yaw_des,
    double yaw_rate_des)
{
    pos_des_ = pos_des;
    vel_des_ = vel_des;
    acc_des_ = acc_des;
    jerk_des_ = jerk_des;
    yaw_des_ = yaw_des;
    yaw_rate_des_ = yaw_rate_des;
}

void TrackingController::UpdateDesired(AircraftControllerDesired desired)
{
    pos_des_ = desired.pos_des;
    vel_des_ = desired.vel_des;
    acc_des_ = desired.acc_des;
    jerk_des_ = desired.jerk_des;
    yaw_des_ = desired.yaw_des;
    yaw_rate_des_ = desired.yaw_rate_des;
}
