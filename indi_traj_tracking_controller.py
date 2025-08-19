import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import signal
import quaternion
from global_dynamics import Environment


class SimpleAircraft:
    def __init__(
        self,
        mass,
        moment_of_inertia,
        flap_length_x,
        flap_length_y,
        thrust_l_y,
        max_elevon_angle,
        max_omega,
        C_l_v,
        C_d_v,
        delta_C_l_v,
        delta_C_l_t,
        C_t,
        C_d_t,
        C_l_t,
        C_u_T,
        C_u,
    ):
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.moment_of_inertia_inv = np.linalg.inv(self.moment_of_inertia)
        self.flap_length_x = (
            flap_length_x  # distance to flap aerocenter from cg along x
        )
        self.flap_length_y = (
            flap_length_y  # distance to flap aerocenter from cg along y
        )
        self.thrust_l_y = thrust_l_y  # distance to prop thrust location from cg along y
        self.max_elevon_angle = max_elevon_angle
        self.max_omega = max_omega
        self.C_l_v = C_l_v  # wing lift coeff
        self.C_d_v = C_d_v  # wing drag coeff
        self.delta_C_l_v = delta_C_l_v  # flap lift due to body airspeed
        self.delta_C_l_t = delta_C_l_t  # flap lift due to prop wash induced airspeed
        self.C_t = C_t  # coefficient of thrust
        self.C_d_t = C_d_t  # drag due to thrust coeff
        self.C_l_t = C_l_t  # lift due to thrust coeff. 0 on symmetrical airoil with no motor angle
        self.C_u_T = C_u_T  # Pitch moment coeff due to thrust. 0 on symmetrical airoil with no motor angle
        self.C_u = C_u  # motor+prop torque coeff


def matrix_cross(vector):
    return np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )


class TrackingController:
    def __init__(
        self,
        pos_gain,
        vel_gain,
        acc_gain,
        quat_gain,
        omega_gain,
        hz,
        aircraft: SimpleAircraft,
        environment: Environment,
    ):
        self.pos_gain = pos_gain
        self.vel_gain = vel_gain
        self.acc_gain = acc_gain
        self.quat_gain = quat_gain
        self.omega_gain = omega_gain
        self.time = 0
        self.hz = hz
        self.dt = 1 / hz
        self.aircraft = aircraft
        self.environment = environment
        self.velocity_body = np.zeros(3)
        self.velocity_magnitude = 0
        self.acceleration_filt = np.zeros(3)
        self.omega_body_filt = np.zeros(3)
        self.omega_body_filt_prev = np.zeros(3)
        self.omega_dot_body_filt = np.zeros(3)
        self.body_to_inertial_rotation = R.from_quat((1, 0, 0, 0), scalar_first=True)
        self.flap_l_filt = 0
        self.flap_r_filt = 0
        self.motor_w_l_filt = 0
        self.motor_w_r_filt = 0

        # saved forces
        self.force_inertial_lpf = np.zeros(3)
        self.force_flaps_body_lpf_hpf = np.zeros(3)
        # saved moments
        self.moment_body_filt = np.zeros(3)
        self.quaternion = np.quaternion(1, 0, 0, 0)

        filter_order = 2
        cutoff_freq = 15
        fs = hz
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff_freq / nyquist
        self.sos_lpf = signal.butter(
            filter_order, normalized_cutoff, btype="low", output="sos"
        )

        self.acceleration_x_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0
        self.acceleration_y_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0
        self.acceleration_z_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0
        self.omega_body_x_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0
        self.omega_body_y_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0
        self.omega_body_z_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0
        self.flap_l_lpf_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0
        self.flap_r_lpf_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0
        self.motor_w_l_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0
        self.motor_w_r_filt_state = signal.sosfilt_zi(self.sos_lpf) * 0

        filter_order = 2
        cutoff_freq = 1
        fs = hz
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff_freq / nyquist
        self.sos_hpf = signal.butter(
            filter_order, normalized_cutoff, btype="high", output="sos"
        )

        self.flap_l_hpf_filt_state = signal.sosfilt_zi(self.sos_hpf) * 0
        self.flap_r_hpf_filt_state = signal.sosfilt_zi(self.sos_hpf) * 0
        self.moment_command = np.array([0, 0.0, 0])

    def update(
        self,
        pos_des,
        vel_des,
        acc_des,
        yaw_des,
        omega_des,
        x,
        u,
        acceleration,
        omega_dot,
    ):
        self.update_estimates(x, x[13:17], acceleration, omega_dot)

        self.time += self.dt
        self.omega_des = omega_des
        self.pos_des = pos_des.copy()
        self.vel_des = vel_des.copy()
        self.acceleration_des = acc_des.copy()
        self.acceleration_command = self.control_position_velocity(
            self.pos_des, self.vel_des, self.acceleration_des
        )
        # self.acceleration_command = [0.1, 0, -0.1]
        self.force_command = self.control_linear_acceleration(self.acceleration_command)
        # self.force_command = -self.force_command
        self.quat_des, self.thrust_des = self.force_yaw_transform(
            self.force_command, yaw_des
        )
        # print(self.thrust_des)
        # self.thrust_des = 1.01 * self.aircraft.mass * 9.81
        # self.quat_des = R.from_euler("ZXY", [0, 0, np.radians(90)]).as_quat(
        #     scalar_first=True
        # )
        # self.omega_des = [0, 0, 0]
        self.omega_dot_des = self.control_attitude_angular_rate(
            self.quat_des, self.omega_des
        )
        self.moment_command = self.control_angular_acceleration(self.omega_dot_des)
        new_u = self.thrust_moment_transform(self.thrust_des, self.moment_command)
        # print(np.degrees(new_u[0]), np.degrees(new_u[1]))
        # print("moment command", self.moment_command, end="\t")
        return new_u

    def get_desired(self):
        return np.concatenate(
            (
                self.quat_des,  # 0, 1, 2, 3
                self.omega_des,  # 4, 5, 6
                self.omega_dot_des,  # 7, 8, 9
                self.pos_des,  # 10, 11, 12
                self.vel_des,  # 13, 14, 15
                self.acceleration_des,
            )
        )

    def get_estimated_moment(self):
        return self.moment_body_filt

    def get_commanded_force_moment(self):
        return self.force_command, self.moment_command

    def get_filtered_data(self):
        return np.concatenate(
            (self.acceleration_filt, self.omega_body_filt, self.omega_dot_body_filt)
        )

    # States x
    ################## Linear #############           ############# Rotational ##########################
    #     6       7       8       9      10       11       12
    #  quat_w, quat_x, quat_y, quat_z, omega_x, omega_y, omega_z

    # Inputs u
    #    0       1         2          3
    # flap_l, flap_r, motor_w_l, motor_w_r
    def update_estimates(self, x, u, acceleration, omega_dot):
        self.velocity_body = np.array([x[3], x[4], x[5]])
        self.velocity_magnitude = np.linalg.norm(self.velocity_body)
        quat_w = x[6]
        quat_x = x[7]
        quat_y = x[8]
        quat_z = x[9]
        self.body_to_inertial_rotation = R.from_quat(
            (quat_w, quat_x, quat_y, quat_z), scalar_first=True
        )
        self.inertial_to_body_rotation = self.body_to_inertial_rotation.inv()
        self.velocity_inertial = self.body_to_inertial_rotation.apply(
            self.velocity_body
        )

        self.quaternion = np.quaternion(quat_w, quat_x, quat_y, quat_z)
        self.position_inertial = x[0:3]
        # self.low_pass_inputs(x[10:13], x[13:17], acceleration)
        self.low_pass_inputs(x[10:13], u, acceleration)

        # self.omega_dot_body_filt = omega_dot.copy()
        self.omega_dot_body_filt = self.hz * (
            self.omega_body_filt - self.omega_body_filt_prev
        )
        self.omega_body_filt_prev = self.omega_body_filt.copy()

        # calculate force estimates

        # thrust
        motor_thrust_l_body_filt = self.aircraft.C_t * self.motor_w_l_filt**2
        motor_thrust_r_body_filt = self.aircraft.C_t * self.motor_w_l_filt**2

        force_thrust_l_body_filt = (1 - self.aircraft.C_d_t) * motor_thrust_l_body_filt
        force_thrust_r_body_filt = (1 - self.aircraft.C_d_t) * motor_thrust_r_body_filt
        force_thrust_body_filt = np.array(
            [force_thrust_l_body_filt + force_thrust_r_body_filt, 0, 0]
        )

        # flaps force
        force_flaps_l_body_lpf_hpf = -self.flap_l_filt * (
            self.aircraft.delta_C_l_t * motor_thrust_l_body_filt
            + self.aircraft.delta_C_l_v
            * self.velocity_magnitude
            * self.velocity_body[0]
        )
        force_flaps_r_body_lpf_hpf = -self.flap_r_filt * (
            self.aircraft.delta_C_l_t * motor_thrust_r_body_filt
            + self.aircraft.delta_C_l_v
            * self.velocity_magnitude
            * self.velocity_body[0]
        )
        self.force_flaps_body_lpf_hpf[2] = (
            force_flaps_l_body_lpf_hpf + force_flaps_r_body_lpf_hpf
        )

        # wing force
        force_wing_body_filt = (
            np.array(
                [
                    -self.aircraft.C_d_v * self.velocity_body[0],
                    0,
                    -self.aircraft.C_l_v * self.velocity_body[2],
                ]
            )
            * self.velocity_magnitude
        )

        force_body_filt = (
            force_thrust_body_filt
            + self.force_flaps_body_lpf_hpf
            + force_wing_body_filt
        )

        # caclulate moments
        # thrust
        thrust_moment_body_filt = np.array(
            [
                0,
                self.aircraft.C_u_T
                * (motor_thrust_l_body_filt + motor_thrust_r_body_filt),
                self.aircraft.thrust_l_y
                * (force_thrust_l_body_filt - force_thrust_r_body_filt),
            ]
        )

        # motor torque
        motor_torque_l_body = -self.aircraft.C_u * self.motor_w_l_filt**2
        motor_torque_r_body = self.aircraft.C_u * self.motor_w_r_filt**2

        motor_torque_moment_body_filt = np.array(
            [motor_torque_l_body + motor_torque_r_body, 0, 0]
        )

        # flap moment
        flap_moment_body = np.array(
            [
                self.aircraft.flap_length_y
                * (force_flaps_r_body_lpf_hpf - force_flaps_l_body_lpf_hpf),
                self.aircraft.flap_length_x
                * (force_flaps_r_body_lpf_hpf + force_flaps_l_body_lpf_hpf),
                0,
            ]
        )
        # wing_moment = matrix_cross(np.array([-0.015, 0, 0])) @ force_wing_body_filt

        self.moment_body_filt = (
            thrust_moment_body_filt
            + motor_torque_moment_body_filt
            + flap_moment_body
            # + wing_moment
        )
        # print(self.moment_body_filt[1], end="\t")
        self.force_inertial_lpf = self.body_to_inertial_rotation.apply(force_body_filt)
        self.acceleration_tilda_lpf = (
            self.acceleration_filt
            - self.body_to_inertial_rotation.apply(self.force_flaps_body_lpf_hpf)
            / self.aircraft.mass
        )

    def dynamics(self, x, u):
        quat_w = x[6]
        quat_x = x[7]
        quat_y = x[8]
        quat_z = x[9]
        velocity_body = np.array([x[3], x[4], x[5]])
        velocity_magnitude = np.linalg.norm(velocity_body)
        body_to_inertial_rotation = R.from_quat(
            (quat_w, quat_x, quat_y, quat_z), scalar_first=True
        )
        inertial_to_body_rotation = body_to_inertial_rotation.inv()
        velocity_inertial = body_to_inertial_rotation.apply(velocity_body)
        omega_x = x[10]
        omega_y = x[11]
        omega_z = x[12]
        omega_body = np.array([omega_x, omega_y, omega_z])
        flap_l = u[0]
        flap_r = u[1]
        motor_w_l = u[2]
        motor_w_r = u[3]
        motor_thrust_l_body_filt = self.aircraft.C_t * motor_w_l**2
        motor_thrust_r_body_filt = self.aircraft.C_t * motor_w_r**2

        force_thrust_l_body_filt = (1 - self.aircraft.C_d_t) * motor_thrust_l_body_filt
        force_thrust_r_body_filt = (1 - self.aircraft.C_d_t) * motor_thrust_r_body_filt
        force_thrust_body_filt = np.array(
            [force_thrust_l_body_filt + force_thrust_r_body_filt, 0, 0]
        )

        # flaps force
        force_flaps_l_body_lpf_hpf = -flap_l * (
            self.aircraft.delta_C_l_t * motor_thrust_l_body_filt
            + self.aircraft.delta_C_l_v * velocity_magnitude * velocity_body[0]
        )
        force_flaps_r_body_lpf_hpf = -flap_r * (
            self.aircraft.delta_C_l_t * motor_thrust_r_body_filt
            + self.aircraft.delta_C_l_v * velocity_magnitude * velocity_body[0]
        )
        self.force_flaps_body_lpf_hpf = np.array(
            [0, 0, force_flaps_l_body_lpf_hpf + force_flaps_r_body_lpf_hpf]
        )

        # wing force
        force_wing_body_filt = (
            np.array(
                [
                    -self.aircraft.C_d_v * velocity_body[0],
                    0,
                    -self.aircraft.C_l_v * velocity_body[2],
                ]
            )
            * velocity_magnitude
        )
        # print(force_wing_body_filt)
        force_gravity_body = self.aircraft.mass * inertial_to_body_rotation.apply(
            np.array([0, 0, self.environment.gravity])
        )
        force_body_filt = (
            force_thrust_body_filt
            + self.force_flaps_body_lpf_hpf
            + force_wing_body_filt
            + force_gravity_body
        )
        # print(
        #     force_body_filt,
        #     force_thrust_body_filt,
        #     self.force_flaps_body_lpf_hpf,
        #     force_wing_body_filt,
        #     force_gravity_body,
        # )
        # caclulate moments
        # thrust
        thrust_moment_body_filt = np.array(
            [
                0,
                self.aircraft.C_u_T
                * (motor_thrust_l_body_filt + motor_thrust_r_body_filt),
                self.aircraft.thrust_l_y
                * (force_thrust_l_body_filt - force_thrust_r_body_filt),
            ]
        )

        # motor torque
        motor_torque_l_body = -self.aircraft.C_u * motor_w_l**2
        motor_torque_r_body = self.aircraft.C_u * motor_w_r**2

        motor_torque_moment_body_filt = np.array(
            [motor_torque_l_body + motor_torque_r_body, 0, 0]
        )

        # flap moment
        flap_moment_body = np.array(
            [
                self.aircraft.flap_length_y
                * (force_flaps_r_body_lpf_hpf - force_flaps_l_body_lpf_hpf),
                self.aircraft.flap_length_x
                * (force_flaps_r_body_lpf_hpf + force_flaps_l_body_lpf_hpf),
                0,
            ]
        )
        # wing_moment = matrix_cross(np.array([-0.015, 0, 0])) @ force_wing_body_filt
        moment_body_filt = (
            thrust_moment_body_filt
            + motor_torque_moment_body_filt
            + flap_moment_body
            # + wing_moment
        )
        # print("moment_body_filt", moment_body_filt)
        acceleration_body = (
            force_body_filt / self.aircraft.mass
            - matrix_cross(omega_body) @ velocity_body
        )

        angular_acceleration_body = (
            self.aircraft.moment_of_inertia_inv @ moment_body_filt
        ) - self.aircraft.moment_of_inertia_inv @ (
            matrix_cross(omega_body) @ self.aircraft.moment_of_inertia @ omega_body
        )

        omega_dot_quat = np.array(
            [
                [0, -omega_x, -omega_y, -omega_z],
                [omega_x, 0, omega_z, -omega_y],
                [omega_y, -omega_z, 0, omega_x],
                [omega_z, omega_y, -omega_x, 0],
            ]
        )

        quat_dot = 0.5 * omega_dot_quat @ np.array([quat_w, quat_x, quat_y, quat_z])

        x_dot = np.zeros(len(x))
        x_dot[0] = velocity_inertial[0]
        x_dot[1] = velocity_inertial[1]
        x_dot[2] = velocity_inertial[2]
        x_dot[3] = acceleration_body[0]
        x_dot[4] = acceleration_body[1]
        x_dot[5] = acceleration_body[2]
        x_dot[6] = quat_dot[0]
        x_dot[7] = quat_dot[1]
        x_dot[8] = quat_dot[2]
        x_dot[9] = quat_dot[3]
        x_dot[10] = angular_acceleration_body[0]
        x_dot[11] = angular_acceleration_body[1]
        x_dot[12] = angular_acceleration_body[2]
        return x_dot

    def low_pass_inputs(self, omega, u, acceleration):
        # acceleration = [acc_body_x, acc_body_y, acc_body_z]
        # need to low pass:
        # acceleration_lpf = (R_body_to_inertial @acceleration_body + gravity)
        # flap deflection
        # motor speeds
        # omega_lpf = omega
        # second order butterworth low pass filter at 15hz
        # second order butterworth high pass filter at 1hz
        acceleration_data_minus_gravity = self.body_to_inertial_rotation.apply(
            acceleration
        )  # + np.array([0, 0, -self.environment.gravity])
        # acceleration_lpf
        self.acceleration_filt = acceleration_data_minus_gravity.copy()
        # self.acceleration_filt[0], self.acceleration_x_filt_state = signal.sosfilt(
        #     self.sos_lpf,
        #     [acceleration_data_minus_gravity[0]],
        #     zi=self.acceleration_x_filt_state,
        # )
        # self.acceleration_filt[1], self.acceleration_y_filt_state = signal.sosfilt(
        #     self.sos_lpf,
        #     [acceleration_data_minus_gravity[1]],
        #     zi=self.acceleration_y_filt_state,
        # )
        # self.acceleration_filt[2], self.acceleration_z_filt_state = signal.sosfilt(
        #     self.sos_lpf,
        #     [acceleration_data_minus_gravity[2]],
        #     zi=self.acceleration_z_filt_state,
        # )

        # omega
        # self.omega_body_filt[0], self.omega_body_x_filt_state = signal.sosfilt(
        #     self.sos_lpf, [omega[0]], zi=self.omega_body_x_filt_state
        # )
        # self.omega_body_filt[1], self.omega_body_y_filt_state = signal.sosfilt(
        #     self.sos_lpf, [omega[1]], zi=self.omega_body_y_filt_state
        # )
        # self.omega_body_filt[2], self.omega_body_z_filt_state = signal.sosfilt(
        #     self.sos_lpf, [omega[2]], zi=self.omega_body_z_filt_state
        # )
        self.omega_body_filt = omega.copy()

        # flap deflection

        # to_fill, self.flap_l_lpf_filt_state = signal.sosfilt(
        #     self.sos_lpf, [u[0]], zi=self.flap_l_lpf_filt_state
        # )
        # to_fill, self.flap_l_hpf_filt_state = signal.sosfilt(
        #     self.sos_hpf, to_fill, zi=self.flap_l_hpf_filt_state
        # )
        # self.flap_l_filt = to_fill[0]

        # to_fill, self.flap_r_lpf_filt_state = signal.sosfilt(
        #     self.sos_lpf, [u[1]], zi=self.flap_r_lpf_filt_state
        # )
        # to_fill, self.flap_r_hpf_filt_state = signal.sosfilt(
        #     self.sos_hpf, to_fill, zi=self.flap_r_hpf_filt_state
        # )
        # self.flap_r_filt = to_fill[0]
        self.flap_l_filt = u[0]
        self.flap_r_filt = u[1]
        # motor speeds
        # to_fill, self.motor_w_l_filt_state = signal.sosfilt(
        #     self.sos_lpf, [u[2]], zi=self.motor_w_l_filt_state
        # )
        # self.motor_w_l_filt = to_fill[0]
        # to_fill, self.motor_w_r_filt_state = signal.sosfilt(
        #     self.sos_lpf, [u[3]], zi=self.motor_w_r_filt_state
        # )
        # self.motor_w_r_filt = to_fill[0]
        self.motor_w_l_filt = u[2]
        self.motor_w_r_filt = u[3]

    def control_position_velocity(self, pos_des, vel_des, acc_des):
        acceleration_command = (
            self.body_to_inertial_rotation.apply(
                self.pos_gain
                @ self.inertial_to_body_rotation.apply(pos_des - self.position_inertial)
                + self.vel_gain
                @ self.inertial_to_body_rotation.apply(vel_des - self.velocity_inertial)
                + self.acc_gain
                @ self.inertial_to_body_rotation.apply(
                    acc_des - self.acceleration_tilda_lpf
                )
            )
            + acc_des
        )
        return acceleration_command

    def control_linear_acceleration(self, acceleration_des):

        force_inertial_commanded = (
            acceleration_des - self.acceleration_tilda_lpf
        ) * self.aircraft.mass + self.force_inertial_lpf

        return force_inertial_commanded

    def force_yaw_transform(self, force_des, yaw_des):
        # return quat and thrust
        # get phi
        inertial_to_yaw_rotation = R.from_euler("ZXY", [yaw_des, 0, 0]).inv()
        beta_x = inertial_to_yaw_rotation.apply(force_des)[1]
        beta_z = force_des[2]
        phi = -np.atan2(beta_x, beta_z)
        inertial_to_phi_rotation = R.from_euler("ZXY", [yaw_des, phi, 0]).inv()
        b_y = self.body_to_inertial_rotation.apply([0, 1, 0])
        phi_y = inertial_to_phi_rotation.inv().apply([0, 1, 0])
        dot = np.dot(b_y, phi_y)
        if dot <= 0:
            phi += np.pi
            inertial_to_phi_rotation = R.from_euler("ZXY", [yaw_des, phi, 0]).inv()
            if phi > np.pi:
                phi = -(np.pi * 2 - phi)
        self.inertial_to_phi_rotation = inertial_to_phi_rotation

        # print(np.degrees(phi), "\t", k, end="\t")
        force_phi = inertial_to_phi_rotation.apply(force_des)
        vel_phi = inertial_to_phi_rotation.apply(self.velocity_inertial)
        delta = (
            self.flap_l_filt + self.flap_r_filt
        )  # decide on which to use here. Either real or commanded
        eta = (-self.aircraft.delta_C_l_t * delta / 2) / (1 - self.aircraft.C_d_t)
        sigma_x = (
            eta
            * (
                force_phi[0]
                + self.aircraft.C_d_v * self.velocity_magnitude * vel_phi[0]
            )
            - self.aircraft.delta_C_l_v * delta * self.velocity_magnitude * vel_phi[0]
            - self.aircraft.C_l_v * self.velocity_magnitude * vel_phi[2]
            - force_phi[2]
        )
        sigma_z = (
            eta
            * (
                force_phi[2]
                + self.aircraft.C_d_v * self.velocity_magnitude * vel_phi[2]
            )
            - self.aircraft.delta_C_l_v * delta * self.velocity_magnitude * vel_phi[2]
            + self.aircraft.C_l_v * self.velocity_magnitude * vel_phi[0]
            + force_phi[0]
        )
        theta = np.atan2(sigma_x, sigma_z)
        # print(np.degrees(theta), end="\t")
        # print(np.degrees(yaw_des))
        # if self.time > 1.744:
        #     print(yaw_des, phi, theta)

        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        thrust = (
            1
            / (1 - self.aircraft.C_d_t)
            * (
                c_theta * force_phi[0]
                - s_theta * force_phi[2]
                + self.aircraft.C_d_v
                * self.velocity_magnitude
                * (c_theta * vel_phi[0] - s_theta * vel_phi[2])
            )
        )
        if thrust < 0:
            theta += np.pi

            c_theta = np.cos(theta)
            s_theta = np.sin(theta)
            thrust = (
                1
                / (1 - self.aircraft.C_d_t)
                * (
                    c_theta * force_phi[0]
                    - s_theta * force_phi[2]
                    + self.aircraft.C_d_v
                    * self.velocity_magnitude
                    * (c_theta * vel_phi[0] - s_theta * vel_phi[2])
                )
            )

        quat = R.from_euler("ZXY", (yaw_des, phi, theta)).as_quat(scalar_first=True)

        return quat, thrust

    def control_attitude_angular_rate(self, quat_des, omega_des):
        quat_des_obj = np.quaternion(quat_des[0], quat_des[1], quat_des[2], quat_des[3])
        error_quat = self.quaternion.conj() * quat_des_obj

        bottom = np.sqrt(1 - error_quat.w**2)
        if self.time > 1.744:
            print(end="")
        if bottom == 0:
            angle_error_vector = np.zeros(3)
            print("uh oh")
        else:
            angle_error_vector = (
                2
                * np.arccos(error_quat.w)
                / bottom
                * np.array([error_quat.x, error_quat.y, error_quat.z])
            )
        # angle_error_vector = np.arctan2(
        #     np.sin(angle_error_vector), np.cos(angle_error_vector)
        # )
        # print(angle_error_vector)
        self.angle_error_vec = angle_error_vector
        # if np.any(np.isnan(angle_error_vector)) or np.any(np.isinf(angle_error_vector)):
        #     angle_error_vector = np.zeros(3)
        omega_dot_commanded = self.quat_gain @ angle_error_vector + self.omega_gain @ (
            omega_des - self.omega_body_filt
        )
        return omega_dot_commanded

    def control_angular_acceleration(self, omega_dot_des):
        moment_commanded = (
            self.aircraft.moment_of_inertia @ (omega_dot_des - self.omega_dot_body_filt)
            + self.moment_body_filt
        )

        # moment_commanded = self.aircraft.moment_of_inertia @ omega_dot_des
        # print("omega dot, moment, moment command")
        # print(
        #     self.omega_dot_body_filt,
        #     self.moment_body_filt,
        #     moment_commanded,
        # )
        # moment_commanded[0] = 0
        # moment_commanded[2] = 0
        return moment_commanded

    def thrust_moment_transform(self, thrust_des, moment_des):
        # thrust = eq 37, 38
        # flap = eq 39
        delta_thrust = moment_des[2] / (
            self.aircraft.thrust_l_y * (1 - self.aircraft.C_d_t)
        )
        thrust_l_body = (thrust_des + delta_thrust) * 0.5
        thrust_r_body = (thrust_des - delta_thrust) * 0.5
        if thrust_l_body < 0:
            thrust_l_body = 0
        if thrust_r_body < 0:
            thrust_r_body = 0
        motor_w_l = -np.sqrt(thrust_l_body / self.aircraft.C_t)
        motor_w_r = np.sqrt(thrust_r_body / self.aircraft.C_t)

        thrust_moment_body_des = np.array(
            [
                0,
                self.aircraft.C_u_T * (thrust_l_body + thrust_r_body),
                self.aircraft.thrust_l_y * (thrust_l_body - thrust_r_body),
            ]
        )

        # motor torque moment estimate
        motor_torque_l_body = -self.aircraft.C_u * motor_w_l**2
        motor_torque_r_body = self.aircraft.C_u * motor_w_r**2
        motor_torque_moment_body_des = np.array(
            [motor_torque_l_body + motor_torque_r_body, 0, 0]
        )

        moment_flaps_des = (
            moment_des - thrust_moment_body_des - motor_torque_moment_body_des
        )

        v_left = (
            -self.aircraft.delta_C_l_t * thrust_l_body
            - self.aircraft.delta_C_l_v
            * self.velocity_magnitude
            * self.velocity_body[0]
        )

        v_right = (
            -self.aircraft.delta_C_l_t * thrust_l_body
            - self.aircraft.delta_C_l_v
            * self.velocity_magnitude
            * self.velocity_body[0]
        )
        flap_pre_1 = np.array(
            [
                [
                    -self.aircraft.flap_length_y * v_left,
                    self.aircraft.flap_length_y * v_right,
                ],
                [
                    self.aircraft.flap_length_x * v_left,
                    self.aircraft.flap_length_x * v_right,
                ],
            ]
        )

        flap_pre_2 = np.array([moment_flaps_des[0], moment_flaps_des[1]])
        flaps = np.linalg.inv(flap_pre_1) @ flap_pre_2
        flap_l = flaps[0]
        flap_r = flaps[1]
        flap_l = np.clip(
            flap_l, -self.aircraft.max_elevon_angle, self.aircraft.max_elevon_angle
        )
        flap_r = np.clip(
            flap_r, -self.aircraft.max_elevon_angle, self.aircraft.max_elevon_angle
        )
        motor_w_l = np.clip(motor_w_l, -self.aircraft.max_omega, 0)
        motor_w_r = np.clip(motor_w_r, 0, self.aircraft.max_omega)
        return [flap_l, flap_r, motor_w_l, motor_w_r]
