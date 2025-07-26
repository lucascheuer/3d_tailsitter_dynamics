from simple_dynamics import *
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from animate import animate_simple
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    mass = 0.2
    moment_of_inertia = np.eye(3) * 0.01
    flap_length_x = 0.075
    flap_length_y = 0.28
    thrust_l_y = 0.275
    C_l_v = 0.17
    C_d_v = 0.01
    delta_C_l_v = 0.041
    delta_C_l_t = 1.7
    C_t = 0.00001
    C_d_t = 0
    C_u_T = 0
    C_u = 0
    chord = 0.15
    wingspan = 0.55
    aircraft = Aircraft(
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
    )
    environment = Environment(9.81, 1.225)
    np.set_printoptions(suppress=True, precision=6)
    quat_0 = R.from_rotvec(np.deg2rad(90) * np.array([0, 1, 0])).as_quat(
        scalar_first=True
    )
    # States x
    ################## Linear #############           ############# Rotational ##########################
    #   0      1      2      3      4       5      6       7       8       9      10       11       12
    # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, quat_w, quat_x, quat_y, quat_z, omega_x, omega_y, omega_z

    # Inputs u
    #    0       1         2          3
    # flap_l, flap_r, motor_w_l, motor_w_r
    state_0 = np.array(
        [0, 0, 0, 0, 0, 0, quat_0[0], quat_0[1], quat_0[2], quat_0[3], 0, 0, 0]
    )
    control_0 = np.array([np.deg2rad(5), np.deg2rad(5), 283, 283])
    t_start = 0
    t_end = 5
    hz = 100.0
    dt = 1 / hz

    t_step_count = int((t_end - t_start) * hz)
    t_steps = np.linspace(t_start, t_end, t_step_count)
    states = np.zeros((13, t_step_count))
    controls = np.zeros((4, t_step_count))
    states[:, 0] = state_0
    controls[:, 0] = control_0
    for step, time_step in zip(range(1, len(t_steps)), t_steps):
        state_0 = states[:, step - 1]
        # if time_step > 0.5 and time_step < 1.5:
        #     control_0[:2] = np.deg2rad(-45)
        # if time_step > 1.5:
        #     control_0[:2] = np.deg2rad(0)
        control_f = control_0
        solution = solve_ivp(
            dynamics,
            [0, dt],
            state_0,
            args=(control_0, aircraft, environment),
            method="RK45",
            rtol=1e-6,
            atol=1e-12,
        )
        state_f = solution.y[:, -1]
        states[:, step] = state_f
        controls[:, step] = control_f

    animate_simple(states, controls, aircraft)

    plt.show()


if __name__ == "__main__":
    main()
