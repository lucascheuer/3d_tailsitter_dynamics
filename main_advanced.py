from global_dynamics import *
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from animate import animate
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    follow = True
    mass = 0.2
    wingspan = 0.55
    chord = 0.15
    depth = 0.01
    moment_of_inertia = np.eye(3)
    moment_of_inertia[0, 0] = 1 / 12 * mass * (wingspan**2 + chord**2)
    moment_of_inertia[1, 1] = 1 / 12 * mass * (depth**2 + chord**2)
    moment_of_inertia[2, 2] = 1 / 12 * mass * (wingspan**2 + depth**2)
    C_t = 0.00001
    C_m = 0  # 0.00001
    S = wingspan * chord  # surface area
    S_p = np.pi * (0.55 * 0.25) ** 2
    C_d_naught = 0.01
    C_y_naught = 0.17
    C_l_p = 0.05  # roll damping
    C_m_q = 0.25  # pitch damping
    C_n_r = 0.005  # yaw damping
    elevon_effectiveness_linear = np.array([0, 0.5, 0])
    elevon_effectiveness_rotational = np.array([0, -0.5, 0])
    elevon_percentage = 0.5

    delta_r = chord * 0.1
    p_l = np.array([0, -wingspan / 4, 0])
    p_r = np.array([0, wingspan / 4, 0])
    # wingspan of 55cm
    aircraft = Aircraft(
        mass,
        moment_of_inertia,
        C_t,
        C_m,
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
        wingspan,
        chord,
        delta_r,
        p_l,
        p_r,
    )
    environment = Environment(9.81, 1.225)
    np.set_printoptions(suppress=True, precision=6)
    orientation_naught = R.from_rotvec(np.deg2rad(0) * np.array([0, 1, 0]))
    quat_0 = orientation_naught.as_quat(scalar_first=True)
    v_body_0 = orientation_naught.apply(np.array([5, 0, 0]))
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
    control_0 = np.array([np.deg2rad(-15), np.deg2rad(-15), 200, 200])
    # initial_state = dynamics(0, state_0, control_0, aircraft, environment, [1])
    # state_0[13:46] = initial_state[13:46]
    t_start = 0
    t_end = 5
    hz = 100.0
    dt = 1 / hz

    t_step_count = int((t_end - t_start) * hz)
    t_steps = np.linspace(t_start, t_end, t_step_count)
    states = np.zeros((13, t_step_count))
    controls = np.zeros((4, t_step_count))
    force_data = np.zeros((36, t_step_count))
    states[:, 0] = state_0
    controls[:, 0] = control_0
    for step, time_step in zip(range(1, len(t_steps)), t_steps):
        state_0 = states[:, step - 1]
        # if time_step > 0.5:  # and time_step < 1.5:
        #     # control_0[:2] = np.deg2rad(0)
        #     control_0[2:] = 0

        control_f = control_0
        solution = solve_ivp(
            dynamics,
            [time_step, time_step + dt],
            state_0,
            args=(control_0, aircraft, environment, force_data[:, step]),
            method="RK45",
            rtol=1e-6,
            atol=1e-12,
        )
        state_f = solution.y[:, -1]
        # print(np.linalg.norm(state_f[28:31]) / aircraft.mass)
        states[:, step] = state_f
        controls[:, step] = control_f

    animate(states, controls, aircraft, force_data=force_data, follow=follow)

    plt.show()


if __name__ == "__main__":
    main()
