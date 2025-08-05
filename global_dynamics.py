import numpy as np
from scipy.spatial.transform import Rotation as R


class Aircraft:
    def __init__(
        self,
        mass,
        moment_of_inertia,
        C_t,
        C_m,
        max_omega,
        S,
        S_p,
        C_d_naught,
        C_y_naught,
        C_l_p,
        C_m_q,
        C_n_r,
        elevon_effectiveness_linear,
        elevon_effectiveness_rotational,
        elevon_percentage,
        max_elevon_angle,
        wingspan,
        chord,
        delta_r,
        p_l,
        p_r,
    ):
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.moment_of_inertia_inv = np.linalg.inv(self.moment_of_inertia)
        self.C_t = C_t  # propeller coefficient of thrust
        self.C_m = C_m  # propeller coefficient of moment
        self.max_omega = max_omega
        self.S = S  # wing surface area?
        self.S_p = S_p  # propeller disc area
        self.C_d_naught = C_d_naught  # minimum drag coefficient. Drag at 0 aoa for symmetrical airfoil
        self.C_y_naught = C_y_naught  # ??? is this max drag coeff? Drag at 90deg?
        self.C_l_p = C_l_p  # drag on roll axis due to roll rate
        self.C_m_q = C_m_q  # drag on pitch axis due to pitch rate
        self.C_n_r = C_n_r  # drag on yaw axis due to yaw rate
        self.phi_m_omega = np.array([[C_l_p, 0, 0], [0, C_m_q, 0], [0, 0, C_n_r]])
        self.phi_f_v = np.array(
            [[C_d_naught, 0, 0], [0, C_y_naught, 0], [0, 0, np.pi * 2 + C_d_naught]]
        )
        self.elevon_effectiveness_linear = elevon_effectiveness_linear  # a 3 vector that defines how the camber changes per axis with elevon angle
        self.elevon_effectiveness_rotational = elevon_effectiveness_rotational
        self.elevon_percentage = elevon_percentage
        self.max_elevon_angle = max_elevon_angle
        self.wingspan = wingspan
        self.chord = chord
        self.B = np.zeros((3, 3))
        self.B[0, 0] = self.wingspan
        self.B[1, 1] = self.chord
        self.B[2, 2] = self.wingspan
        self.delta_r = delta_r  # distance from aero center to cg. Positive with aero center behind cg
        self.phi_m_v = np.array(
            [
                [0, 0, 0],
                [0, 0, -(1 / chord) * delta_r * (np.pi * 2 + C_d_naught)],
                [0, (1 / wingspan) * delta_r * C_y_naught, 0],
            ]
        )
        self.p_l = p_l  # left propeller location (3 vector)
        self.p_r = p_r  # left propeller location (3 vector)
        self.a_l = np.array(
            [-self.delta_r, -wingspan * 0.25, 0.0]
        )  # vector from cg that aerodynamic forces are applied on the left side
        self.a_r = np.array(
            [-self.delta_r, wingspan * 0.25, 0.0]
        )  # vector from cg that aerodynamic forces are applied on the right side


class Environment:
    def __init__(self, gravity, air):
        self.gravity = gravity
        self.air = air


def matrix_cross(vector):
    return np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )


# States x
################## Linear #############           ############# Rotational ##########################
#   0      1      2      3      4       5      6       7       8       9      10       11       12
# pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, quat_w, quat_x, quat_y, quat_z, omega_x, omega_y, omega_z

# Extra States
################## Linear #############           ############# Rotational ##########################
#   13:16                     16:19                     19:22                        22:25            25:28            28:31                     31:34                    34:37                    37:40                                  40:43                      43:46
# motor_thrust_l_body, motor_thrust_r_body, body_propeller_drag_l_body, body_propeller_drag_r_body, lift_body, force_gravity_body, elevon_lift_reduction_body, rotational_lift_wing, elevon_thrust_redirection_l_body, elevon_thrust_redirection_r_body, rotational_lift_reduction_elevons


# Inputs u
#    0       1         2          3
# flap_l, flap_r, motor_w_l, motor_w_r
def dynamics(
    t,
    x,
    u,
    aircraft: Aircraft,
    environment: Environment,
    get_force_data=False,
    force_data: np.array = np.zeros(0),
):
    # print(force_data)

    pos_x = x[0]
    pos_y = x[1]
    pos_z = x[2]

    velocity_body = np.array([x[3], x[4], x[5]])
    vel_magnitude = np.linalg.norm(velocity_body)
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
    velocity_inertial = body_to_inertial_rotation.apply(velocity_body)
    omega_x = x[10]
    omega_y = x[11]
    omega_z = x[12]
    omega_body = np.array([omega_x, omega_y, omega_z])

    flap_l = u[0]
    flap_r = u[1]
    motor_w_l = u[2]
    motor_w_r = u[3]
    flap_l = np.clip(flap_l, -aircraft.max_elevon_angle, aircraft.max_elevon_angle)
    flap_r = np.clip(flap_r, -aircraft.max_elevon_angle, aircraft.max_elevon_angle)
    motor_w_l = np.clip(motor_w_l, -aircraft.max_omega, 0)
    motor_w_r = np.clip(motor_w_r, 0, aircraft.max_omega)

    ##### Forces ####
    # Gravity
    force_gravity_body = aircraft.mass * inertial_to_body_rotation.apply(
        np.array([0, 0, environment.gravity])
    )
    # print(np.linalg.norm(force_gravity_body))

    # Force motor thrust (x)
    motor_thrust_l_body = np.array(
        [aircraft.C_t * environment.air * motor_w_l**2, 0, 0]
    )
    motor_thrust_r_body = np.array(
        [aircraft.C_t * environment.air * motor_w_r**2, 0, 0]
    )

    # Drag on the body due to propeller thrust (basically propeller loss)
    propeller_drag_l_body = (
        -aircraft.S / (4 * aircraft.S_p) * aircraft.phi_f_v @ motor_thrust_l_body
    )
    propeller_drag_r_body = (
        -0.25 * aircraft.S / (aircraft.S_p) * aircraft.phi_f_v @ motor_thrust_r_body
    )

    # Lift on the body as if it were its original airfoil (non-modified or cambered)
    lift_body = (
        -0.5
        * environment.air
        * aircraft.S
        * aircraft.phi_f_v
        @ velocity_body
        * vel_magnitude
    )

    # Reduction in lift due to flap deflection
    elevon_lift_reduction_l_body = (
        0.25
        * environment.air
        * aircraft.S
        * aircraft.phi_f_v
        @ matrix_cross(aircraft.elevon_effectiveness_linear)
        * flap_l
        * vel_magnitude
        @ velocity_body
    )

    elevon_lift_reduction_r_body = (
        0.25
        * environment.air
        * aircraft.S
        * aircraft.phi_f_v
        @ matrix_cross(aircraft.elevon_effectiveness_linear)
        * flap_r
        * vel_magnitude
        @ velocity_body
    )
    elevon_lift_reduction_body = (
        elevon_lift_reduction_l_body + elevon_lift_reduction_r_body
    )
    # Lift on the body due to the body rotating in the air
    rotational_lift_wing = (
        -0.5
        * environment.air
        * aircraft.S
        * vel_magnitude
        * aircraft.phi_m_v
        @ aircraft.B
        @ omega_body
    )

    # Force caused by the elevon redirecting thrust
    elevon_thrust_redirection_l_body = (
        0.25
        * aircraft.S
        / aircraft.S_p
        * aircraft.phi_f_v
        @ matrix_cross(aircraft.elevon_effectiveness_linear)
        * flap_l
        @ motor_thrust_l_body
    )
    elevon_thrust_redirection_r_body = (
        0.25
        * aircraft.S
        / aircraft.S_p
        * aircraft.phi_f_v
        @ matrix_cross(aircraft.elevon_effectiveness_linear)
        * flap_r
        @ motor_thrust_r_body
    )

    # Reduction in rotational lift due to elevons
    rotational_elevon_lift_reduction_l_body = (
        0.25
        * environment.air
        * aircraft.S
        * aircraft.phi_m_v
        @ matrix_cross(aircraft.elevon_effectiveness_linear)
        * flap_l
        * vel_magnitude
        @ aircraft.B
        @ omega_body
    )

    rotational_elevon_lift_reduction_r_body = (
        0.25
        * environment.air
        * aircraft.S
        * aircraft.phi_m_v
        @ matrix_cross(aircraft.elevon_effectiveness_linear)
        * flap_r
        * vel_magnitude
        @ aircraft.B
        @ omega_body
    )

    rotational_elevon_lift_reduction_body = (
        rotational_elevon_lift_reduction_l_body
        + rotational_elevon_lift_reduction_r_body
    )
    # total force
    total_force_body = (
        motor_thrust_l_body
        + motor_thrust_r_body
        + propeller_drag_l_body
        + propeller_drag_r_body
        + lift_body
        + force_gravity_body
        + elevon_lift_reduction_l_body
        + elevon_lift_reduction_r_body
        + rotational_lift_wing
        + elevon_thrust_redirection_l_body
        + elevon_thrust_redirection_r_body
        + rotational_elevon_lift_reduction_body
    )

    acceleration_body = (
        total_force_body / aircraft.mass - matrix_cross(omega_body) @ velocity_body
    )

    ##### Moments #####
    # moments from forces

    motor_thrust_moment_l_body = matrix_cross(aircraft.p_l) @ motor_thrust_l_body
    motor_thrust_moment_r_body = matrix_cross(aircraft.p_r) @ motor_thrust_r_body

    propeller_drag_moment_l_body = matrix_cross(aircraft.a_l) @ propeller_drag_l_body
    propeller_drag_moment_r_body = matrix_cross(aircraft.a_r) @ propeller_drag_r_body
    elevon_thrust_redirection_moment_l_body = (
        matrix_cross(aircraft.a_l) @ elevon_thrust_redirection_l_body
    )
    elevon_thrust_redirection_moment_r_body = (
        matrix_cross(aircraft.a_r) @ elevon_thrust_redirection_r_body
    )
    elevon_lift_reduction_moment_l_body = (
        matrix_cross(aircraft.a_l) @ elevon_lift_reduction_l_body
    )
    elevon_lift_reduction_moment_r_body = (
        matrix_cross(aircraft.a_r) @ elevon_lift_reduction_r_body
    )
    rotational_elevon_lift_reduction_moment_l_body = (
        matrix_cross(aircraft.a_l) @ rotational_elevon_lift_reduction_l_body
    )
    rotational_elevon_lift_reduction_moment_r_body = (
        matrix_cross(aircraft.a_r) @ rotational_elevon_lift_reduction_r_body
    )

    # Motor Torque Moments
    motor_torque_l_body = np.array(
        [-np.sign(motor_w_l) * aircraft.C_m * motor_w_l**2, 0, 0]
    )
    motor_torque_r_body = np.array(
        [-np.sign(motor_w_r) * aircraft.C_m * motor_w_r**2, 0, 0]
    )
    # moments from moment coefficients
    # phi_mv moments
    lift_moment_body = (
        0.5
        * environment.air
        * aircraft.S
        * aircraft.B
        @ aircraft.phi_m_v
        * vel_magnitude
        @ velocity_body
    )
    propeller_drag_coeff_moment_l_body = (
        -0.25
        * aircraft.S
        * aircraft.S_p
        * aircraft.B
        @ aircraft.phi_m_v
        @ motor_thrust_l_body
    )
    propeller_drag_coeff_moment_r_body = (
        -0.25
        * aircraft.S
        * aircraft.S_p
        * aircraft.B
        @ aircraft.phi_m_v
        @ motor_thrust_r_body
    )
    elevon_lift_reduction_coeff_moment_body = (
        0.25
        * environment.air
        * aircraft.S
        * aircraft.B
        @ aircraft.phi_m_v
        @ matrix_cross(aircraft.elevon_effectiveness_rotational)
        * (flap_l + flap_r)
        * vel_magnitude
        @ velocity_body
    )

    elevon_thrust_redirection_coeff_moment_l_body = (
        0.25
        * aircraft.S
        / aircraft.S_p
        * aircraft.B
        @ aircraft.phi_m_v
        @ matrix_cross(aircraft.elevon_effectiveness_rotational)
        * flap_l
        @ motor_thrust_l_body
    )
    elevon_thrust_redirection_coeff_moment_r_body = (
        0.25
        * aircraft.S
        / aircraft.S_p
        * aircraft.B
        @ aircraft.phi_m_v
        @ matrix_cross(aircraft.elevon_effectiveness_rotational)
        * flap_r
        @ motor_thrust_r_body
    )
    # phi_m_omega moments
    lift_moment_damping_body = (
        -0.5
        * environment.air
        * aircraft.S
        * aircraft.B
        @ aircraft.phi_m_omega
        * vel_magnitude
        @ aircraft.B
        @ omega_body
    )
    elevon_lift_reduction_moment_damping_body = (
        0.25
        * environment.air
        * aircraft.S
        * aircraft.B
        @ aircraft.phi_m_omega
        @ matrix_cross(aircraft.elevon_effectiveness_rotational)
        * (flap_l + flap_r)
        * vel_magnitude
        @ aircraft.B
        @ omega_body
    )

    total_moment_body = (
        motor_thrust_moment_l_body
        + motor_thrust_moment_r_body
        + propeller_drag_moment_l_body
        + propeller_drag_moment_r_body
        + elevon_thrust_redirection_moment_l_body
        + elevon_thrust_redirection_moment_r_body
        + elevon_lift_reduction_moment_l_body
        + elevon_lift_reduction_moment_r_body
        + rotational_elevon_lift_reduction_moment_l_body
        + rotational_elevon_lift_reduction_moment_r_body
        + motor_torque_l_body
        + motor_torque_r_body
        + lift_moment_body  # phi_mv stuf
        + propeller_drag_coeff_moment_l_body
        + propeller_drag_coeff_moment_r_body
        + elevon_lift_reduction_coeff_moment_body
        + elevon_thrust_redirection_coeff_moment_l_body
        + elevon_thrust_redirection_coeff_moment_r_body
        + lift_moment_damping_body  # phi_m_omega stuff
        + elevon_lift_reduction_moment_damping_body
    )
    angular_acceleration_body = (
        aircraft.moment_of_inertia_inv @ total_moment_body
    ) - aircraft.moment_of_inertia_inv @ (
        matrix_cross(omega_body) @ aircraft.moment_of_inertia @ omega_body
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
    # print(total_moment_body[1], omega[1], angular_acceleration[1])
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

    if get_force_data:
        force_data[0:3] = motor_thrust_l_body
        force_data[3:6] = motor_thrust_r_body
        force_data[6:9] = propeller_drag_l_body
        force_data[9:12] = propeller_drag_r_body
        force_data[12:15] = lift_body
        force_data[15:18] = force_gravity_body
        force_data[18:21] = elevon_lift_reduction_l_body
        force_data[21:24] = elevon_lift_reduction_r_body
        force_data[24:27] = rotational_lift_wing
        force_data[27:30] = elevon_thrust_redirection_l_body
        force_data[30:33] = elevon_thrust_redirection_r_body
        force_data[33:36] = rotational_elevon_lift_reduction_body
    return x_dot
