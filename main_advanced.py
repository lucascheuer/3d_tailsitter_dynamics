from global_dynamics import *
from indi_traj_tracking_controller import *
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from animate import animate
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    follow = True
    trail = True
    hz = 100.0
    rho = 1.225
    gravity = 9.81
    environment = Environment(gravity, rho)

    mass = 0.2
    wingspan = 0.55
    chord = 0.15
    depth = 0.01
    moment_of_inertia = np.eye(3)
    moment_of_inertia[0, 0] = 1 / 12 * mass * (wingspan**2 + chord**2)
    moment_of_inertia[1, 1] = 1 / 12 * mass * (depth**2 + chord**2)
    moment_of_inertia[2, 2] = 1 / 12 * mass * (wingspan**2 + depth**2)
    C_t = 0.00001
    C_m = 0.00000001
    S = wingspan * chord  # surface area
    S_p = np.pi * (wingspan * 0.25) ** 2  # each prop is span in diameter
    C_d_naught = 0.01 / rho
    C_y_naught = 0.17 / rho
    C_l_p = 0.05  # roll damping
    C_m_q = 0.025  # pitch damping
    C_n_r = 0.005  # yaw damping
    elevon_effectiveness_linear = np.array([0, 0.5, 0])
    elevon_effectiveness_rotational = np.array([0, -0.5, 0])
    elevon_percentage = 0.5
    max_elevon_angle = np.deg2rad(30)
    max_omega = 400
    delta_r = chord * 0.1
    p_l = np.array([0, -wingspan / 4, 0])
    p_r = np.array([0, wingspan / 4, 0])
    # wingspan of 55cm
    aircraft = Aircraft(
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
    )
    flap_length_x = np.abs(p_l[1])
    flap_length_y = np.abs(delta_r + 0.5 * chord * (1 - elevon_percentage))
    delta_C_l_v = 0.041
    delta_C_l_t = 1.7
    C_d_t = 0
    C_u_T = 0
    simple_aircraft = SimpleAircraft(
        mass,
        moment_of_inertia,
        flap_length_x,
        flap_length_y,
        flap_length_x,
        max_elevon_angle,
        max_omega,
        C_y_naught * rho,
        C_d_naught * rho,
        delta_C_l_v,
        delta_C_l_t,
        C_t * rho,
        C_d_t,
        0,
        C_u_T,
        C_m * rho,
    )
    angular_acceleration_controller = TrackingController(
        hz, simple_aircraft, environment
    )
    np.set_printoptions(suppress=True, precision=6)
    orientation_naught = R.from_euler(
        "XYZ", [np.deg2rad(0), np.deg2rad(90), np.deg2rad(0)]
    )
    quat_0 = orientation_naught.as_quat(scalar_first=True)
    v_body_0 = orientation_naught.apply(np.array([0, 0, 0]))
    # States x
    ################## Linear #############           ############# Rotational ##########################
    #   0      1      2      3      4       5      6       7       8       9      10       11       12
    # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, quat_w, quat_x, quat_y, quat_z, omega_x, omega_y, omega_z

    # Inputs u
    #    0       1         2          3
    # flap_l, flap_r, motor_w_l, motor_w_r
    state_0 = np.array(
        [
            0,
            0,
            0,
            v_body_0[0],
            v_body_0[1],
            v_body_0[2],
            quat_0[0],
            quat_0[1],
            quat_0[2],
            quat_0[3],
            0,
            0,
            0,
        ]
    )
    control_0 = np.array([np.deg2rad(0), np.deg2rad(0), -283, 283])
    # initial_state = dynamics(0, state_0, control_0, aircraft, environment, [1])
    # state_0[13:46] = initial_state[13:46]

    omega_dot_des_body = np.array([0, 0, 0])
    t_start = 0
    t_end = 5
    dt = 1 / hz
    t_step_count = int((t_end - t_start) * hz)
    t_steps = np.linspace(t_start, t_end, t_step_count)
    states = np.zeros((13, t_step_count))
    accelerations = np.zeros((3, t_step_count))
    controls = np.zeros((4, t_step_count))
    force_data = np.zeros((36, t_step_count))
    states[:, 0] = state_0
    controls[:, 0] = control_0
    control_f = control_0
    for step, time_step in zip(range(1, len(t_steps)), t_steps):
        state_0 = states[:, step - 1]
        # control_f = control_0
        real_acceleration = dynamics(0, state_0, control_f, aircraft, environment)
        real_acceleration = np.array(
            [real_acceleration[3], real_acceleration[4], real_acceleration[5]]
        )
        solution = solve_ivp(
            dynamics,
            [time_step, time_step + dt],
            state_0,
            args=(control_f, aircraft, environment, True, force_data[:, step]),
            method="RK45",
            rtol=1e-6,
            atol=1e-12,
        )
        state_f = solution.y[:, -1]
        states[:, step] = state_f
        controls[:, step] = control_f
        accelerations[:, step] = real_acceleration
        control_f = angular_acceleration_controller.update(
            gravity * mass * 1,
            omega_dot_des_body,
            state_f,
            control_f,
            real_acceleration,
        )
        # print(control_test)
        # print(np.linalg.norm(state_f[28:31]) / aircraft.mass)

    animate(
        states, controls, aircraft, force_data=force_data, follow=follow, trail=trail
    )

    plt.show()
    plt.figure()
    plt.plot(t_steps, controls[0, :])
    plt.plot(t_steps, controls[1, :])
    plt.figure()
    plt.plot(t_steps, controls[2, :])
    plt.plot(t_steps, controls[3, :])
    plt.show()


if __name__ == "__main__":
    main()
