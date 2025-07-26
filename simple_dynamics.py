import numpy as np
from scipy.spatial.transform import Rotation as R


#####
# These dynamics don't take into account center of pressure vs center of mass so velocity doesn't create moments on the vehicle. It's weird.
class Aircraft:
    def __init__(
        self,
        mass,
        moment_of_inertia,
        flap_length_x,
        flap_length_y,
        thrust_l_y,
        C_l_v,
        C_d_v,
        delta_C_l_v,
        delta_C_l_t,
        C_t,
        C_d_t,
        C_u_T,
        C_u,
        chord,
        wingspan,
    ):
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.flap_length_x = flap_length_x
        self.flap_length_y = flap_length_y
        self.thrust_l_y = thrust_l_y
        self.C_l_v = C_l_v
        self.C_d_v = C_d_v
        self.delta_C_l_v = delta_C_l_v
        self.delta_C_l_t = delta_C_l_t
        self.C_t = C_t
        self.C_d_t = C_d_t
        self.C_u_T = C_u_T
        self.C_u = C_u
        self.chord = chord
        self.wingspan = wingspan


class Environment:
    def __init__(self, gravity, air):
        self.gravity = gravity
        self.air = air


# States x
################## Linear #############           ############# Rotational ##########################
#   0      1      2      3      4       5      6       7       8       9      10       11       12
# pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, quat_w, quat_x, quat_y, quat_z, omega_x, omega_y, omega_z


# Inputs u
#    0       1         2          3
# flap_l, flap_r, motor_w_l, motor_w_r
def dynamics(t, x, u, aircraft: Aircraft, environment: Environment):
    pos_x = x[0]
    pos_y = x[1]
    pos_z = x[2]

    velocity_inertial = np.array([x[3], x[4], x[5]])
    vel_magnitude = np.linalg.norm(velocity_inertial)
    quat_w = x[6]
    quat_x = x[7]
    quat_y = x[8]
    quat_z = x[9]
    # rotation from inertial frame to alpha/body frame
    body_to_inertial_rotation = R.from_quat(
        (quat_w, quat_x, quat_y, quat_z), scalar_first=True
    )
    # rotation from alpha/body frame to inertial frame
    inertial_to_body_rotation = body_to_inertial_rotation.inv()
    omega_x = x[10]
    omega_y = x[11]
    omega_z = x[12]
    omega = np.array([omega_x, omega_y, omega_z])

    flap_l = u[0]
    flap_r = u[1]
    motor_w_l = u[2]
    motor_w_r = u[3]

    velocity_body = inertial_to_body_rotation.apply(velocity_inertial)
    vel_x_body = velocity_body[0]
    vel_y_body = velocity_body[1]
    vel_z_body = velocity_body[2]
    # Accelerations/forces
    acceleration_gravity_inertial = np.array([0, 0, environment.gravity])

    # Force motor thrust (x)
    motor_thrust_l_body = aircraft.C_t * environment.air * motor_w_l**2
    motor_thrust_r_body = aircraft.C_t * environment.air * motor_w_r**2
    # Actual thrust force due to wing behind motor (x)
    wing_drag_thrust_l_body = (
        1 - aircraft.C_d_t * environment.air
    ) * motor_thrust_l_body
    wing_drag_thrust_r_body = (
        1 - aircraft.C_d_t * environment.air
    ) * motor_thrust_r_body

    thrust_force_body = np.array(
        [wing_drag_thrust_l_body + wing_drag_thrust_r_body, 0, 0]
    )
    # Flap force (z)
    flap_force_l_body = (
        -aircraft.delta_C_l_t * environment.air * motor_thrust_l_body
        - aircraft.delta_C_l_v * environment.air * vel_magnitude * vel_x_body
    ) * flap_l
    flap_force_r_body = (
        -aircraft.delta_C_l_t * environment.air * motor_thrust_r_body
        - aircraft.delta_C_l_v * environment.air * vel_magnitude * vel_x_body
    ) * flap_r

    flap_force_body = np.array([0, 0, flap_force_l_body + flap_force_r_body])

    # Wing force
    wing_force_body = (
        -np.array([aircraft.C_d_v * vel_x_body, 0, aircraft.C_l_v * vel_z_body])
        * environment.air
        * vel_magnitude
    )

    # total force
    total_force_body = thrust_force_body + flap_force_body + wing_force_body
    # print(body_to_inertial_rotation.apply(total_force_body) / aircraft.mass)
    # total acceleration inertial frame
    acceleration = (
        acceleration_gravity_inertial
        + (body_to_inertial_rotation.apply(total_force_body)) / aircraft.mass
    )
    # if np.isclose(0, t):
    #     print(
    #         thrust_force_body / aircraft.mass,
    #         # flap_force_body / aircraft.mass,
    #         wing_force_body / aircraft.mass,
    #         body_to_inertial_rotation.apply(total_force_body) / aircraft.mass,
    #         acceleration,
    #     )
    omega_dot_quat = np.array(
        [
            [0, -omega_x, -omega_y, -omega_z],
            [omega_x, 0, omega_z, -omega_y],
            [omega_y, -omega_z, 0, omega_x],
            [omega_z, omega_y, -omega_x, 0],
        ]
    )
    quat_dot = 0.5 * omega_dot_quat @ np.array([quat_w, quat_x, quat_y, quat_z])
    # Moments
    thrust_moment_body = np.array(
        [
            0,
            aircraft.C_u_T * (motor_thrust_l_body + motor_thrust_r_body),
            aircraft.thrust_l_y * (wing_drag_thrust_l_body - wing_drag_thrust_r_body),
        ]
    )

    motor_torque_l_body = -aircraft.C_u * motor_w_l**2
    motor_torque_r_body = aircraft.C_u * motor_w_r**2

    motor_torque_moment_body = np.array(
        [motor_torque_l_body + motor_torque_r_body, 0, 0]
    )

    flap_moment_body = np.array(
        [
            aircraft.flap_length_y * (flap_force_r_body - flap_force_l_body),
            aircraft.flap_length_x * (flap_force_r_body + flap_force_l_body),
            0,
        ]
    )

    total_moment_body = thrust_moment_body + motor_torque_moment_body + flap_moment_body

    angular_acceleration = np.linalg.inv(aircraft.moment_of_inertia) @ (
        total_moment_body - np.cross(omega, aircraft.moment_of_inertia @ omega)
    )
    # print(total_moment_body[1], omega[1], angular_acceleration[1])
    x_dot = np.zeros(len(x))
    # x_dot[0] = vel_x
    # x_dot[1] = vel_y
    # x_dot[2] = vel_z
    x_dot[0] = velocity_inertial[0]
    x_dot[1] = velocity_inertial[1]
    x_dot[2] = velocity_inertial[2]
    x_dot[3] = acceleration[0]
    x_dot[4] = acceleration[1]
    x_dot[5] = acceleration[2]
    x_dot[6] = quat_dot[0]
    x_dot[7] = quat_dot[1]
    x_dot[8] = quat_dot[2]
    x_dot[9] = quat_dot[3]
    x_dot[10] = angular_acceleration[0]
    x_dot[11] = angular_acceleration[1]
    x_dot[12] = angular_acceleration[2]

    return x_dot
