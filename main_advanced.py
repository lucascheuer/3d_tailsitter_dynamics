from global_dynamics import *
from indi_traj_tracking_controller import *
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from animate import animate
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


def main():
    follow = False
    trail = True
    des_trail = True
    hz = 1000.0
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
    C_d_naught = 0.01 / rho / S / 0.5
    C_y_naught = 0.17 / rho / S / 0.5
    C_l_p = 0.05  # roll damping
    C_m_q = 0.025  # pitch damping
    C_n_r = 0.005  # yaw damping
    elevon_effectiveness_linear = np.array([0, 0.33, 0])
    elevon_effectiveness_rotational = np.array([0, -1.69, 0])
    elevon_percentage = 0.5
    max_elevon_angle = np.deg2rad(30)
    max_elevon_dot = np.deg2rad(1200)  # rad/s
    max_omega = 400
    max_omega_dot = 1600  # max_omega / 0.025  # rad/s/s
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
        max_omega_dot,
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
        max_elevon_dot,
        wingspan,
        chord,
        delta_r,
        p_l,
        p_r,
    )
    flap_length_y = np.abs(p_l[1])
    elevon_chord = chord * elevon_percentage
    chord_no_elevon = chord - elevon_chord
    flap_length_x = np.abs(
        chord_no_elevon + 0.5 * elevon_chord - 0.25 * chord + delta_r
    )
    delta_C_l_v = 0.055
    delta_C_l_t = 0.75
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
        C_y_naught * rho * S * 0.5,
        C_d_naught * rho * S * 0.5,
        delta_C_l_v,
        delta_C_l_t,
        C_t * rho,
        C_d_t,
        0,
        C_u_T,
        C_m * rho,
    )

    pos_gain = np.zeros((3, 3))
    pos_gain[0, 0] = 30
    pos_gain[1, 1] = 0
    pos_gain[2, 2] = 15
    vel_gain = np.zeros((3, 3))
    vel_gain[0, 0] = 25
    vel_gain[1, 1] = 1
    vel_gain[2, 2] = 12
    acc_gain = np.zeros((3, 3))
    acc_gain[0, 0] = 1
    acc_gain[1, 1] = 1
    acc_gain[2, 2] = 1
    quat_gain = np.zeros((3, 3))
    quat_gain[0, 0] = 75  # roll
    quat_gain[1, 1] = 100  # pitch
    quat_gain[2, 2] = 100  # yaw
    omega_gain = np.zeros((3, 3))
    omega_gain[0, 0] = 15  # roll
    omega_gain[1, 1] = 100  # pitch
    omega_gain[2, 2] = 20  # yaw
    angular_acceleration_controller = TrackingController(
        pos_gain,
        vel_gain,
        acc_gain,
        quat_gain,
        omega_gain,
        hz,
        simple_aircraft,
        environment,
    )
    np.set_printoptions(suppress=True, precision=6)
    orientation_naught = R.from_euler(
        "ZXY", [np.deg2rad(0), np.deg2rad(0), np.deg2rad(90)]
    )
    quat_0 = orientation_naught.as_quat(scalar_first=True)
    traj_base_amp = 2.5
    traj_base_freq = 0.125
    v_body_0 = np.array([0, 0, traj_base_amp * np.pi * 2 * traj_base_freq])
    # States x
    ################## Linear #############           ############# Rotational ##########################
    #   0      1      2      3      4       5      6       7       8       9      10       11       12
    # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, quat_w, quat_x, quat_y, quat_z, omega_x, omega_y, omega_z

    # Inputs u
    #    0       1         2          3
    # flap_l, flap_r, motor_w_l, motor_w_r
    control_0 = np.array([np.deg2rad(-0), np.deg2rad(-0), -293, 293])
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
            control_0[0],
            control_0[1],
            control_0[2],
            control_0[3],
        ]
    )

    # initial_state = dynamics(0, state_0, control_0, aircraft, environment, [1])
    # state_0[13:46] = initial_state[13:46]

    t_start = 0
    t_end = 10
    dt = 1 / hz
    t_step_count = int((t_end - t_start) * hz)
    t_steps = np.linspace(t_start, t_end, t_step_count)
    states = np.zeros((17, t_step_count))
    positions_des = np.zeros((3, t_step_count))
    velocities_des = np.zeros((3, t_step_count))
    velocities_inertial = np.zeros((3, t_step_count))
    accelerations = np.zeros((3, t_step_count))
    accelerations_filtered = np.zeros((3, t_step_count))
    accelerations_des = np.zeros((3, t_step_count))
    force_commanded = np.zeros((3, t_step_count))
    force_achieved = np.zeros((3, t_step_count))
    moment_commanded = np.zeros((3, t_step_count))
    moment_achieved = np.zeros((3, t_step_count))
    euler_des = np.zeros((3, t_step_count))
    omega_filtered = np.zeros((3, t_step_count))
    omega_des = np.zeros((3, t_step_count))
    omega_dot = np.zeros((3, t_step_count))
    omega_dot_filtered = np.zeros((3, t_step_count))
    omega_dot_des = np.zeros((3, t_step_count))
    controls = np.zeros((4, t_step_count))
    controls_des = np.zeros((4, t_step_count))
    orientation = np.zeros((3, t_step_count))
    force_data = np.zeros((36, t_step_count))
    angle_error_vec = np.zeros((3, t_step_count))
    phi_data = []

    states[:, 0] = state_0
    controls[:, 0] = control_0
    control_f = control_0
    state_f = state_0
    last_percentage = 0
    print_debug = False
    for step, time_step in zip(range(1, len(t_steps)), t_steps):
        percentage = time_step / (t_end - t_start)
        # print(percentage * 100)

        if percentage >= last_percentage + 0.1:
            print(int(percentage * 100))
            last_percentage = percentage
        # print_debug = True
        state_0 = states[:, step - 1]
        # control_f = control_0
        accel_data = dynamics(
            0,
            state_0,
            control_f,
            aircraft,
            environment,
            print_debug=print_debug,
            est_mom=angular_acceleration_controller.get_estimated_moment(),
        )
        # print_debug = False
        # print()
        omega_dot[:, step] = accel_data[10:13]
        moment_achieved[:, step] = moment_of_inertia @ omega_dot[:, step]
        real_acceleration = np.array([accel_data[3], accel_data[4], accel_data[5]])
        control_dot = [
            flaps(control_f[0], state_f[13], dt),
            flaps(control_f[1], state_f[14], dt),
            motors(control_f[2], state_f[15], dt),
            motors(control_f[3], state_f[16], dt),
        ]
        # control_dot = [0, 0, 0, 0]
        solution = solve_ivp(
            dynamics,
            [time_step, time_step + dt],
            state_0,
            args=(control_dot, aircraft, environment, True, force_data[:, step]),
            method="RK45",
            rtol=1e-6,
            atol=1e-12,
        )
        state_f = solution.y[:, -1]
        states[:, step] = state_f
        controls_des[:, step] = control_f
        controls[:, step] = state_f[13:17]
        body_to_inertial_rotation = R.from_quat(
            (state_f[6], state_f[7], state_f[8], state_f[9]), scalar_first=True
        )
        orientation[:, step] = body_to_inertial_rotation.as_euler("ZXY")
        velocities_inertial[:, step] = body_to_inertial_rotation.apply(state_f[3:6])
        accelerations[:, step] = body_to_inertial_rotation.apply(real_acceleration)
        force_achieved[:, step] = (
            accelerations[:, step] + np.array([0, 0, -9.81])
        ) * mass
        pos_x = traj_base_amp * np.sin(np.pi * 2.0 * traj_base_freq * time_step)
        vel_x = (
            traj_base_amp
            * np.pi
            * 2.0
            * traj_base_freq
            * np.cos(np.pi * 2.0 * traj_base_freq * time_step)
        )
        acc_x = (
            -traj_base_amp
            * np.pi
            * np.pi
            * 2.0
            * 2.0
            * traj_base_freq
            * traj_base_freq
            * np.sin(np.pi * 2.0 * traj_base_freq * time_step)
        )

        control_f = angular_acceleration_controller.update(
            [pos_x, 0, 0],
            [vel_x, 0, 0],
            [acc_x, 0, 0],
            0,
            [0, 0, 0],
            state_f,
            control_f,
            real_acceleration,
            omega_dot[:, step],
        )
        phi_data.append(
            copy.deepcopy(angular_acceleration_controller.inertial_to_phi_rotation)
        )
        filtered_data = angular_acceleration_controller.get_filtered_data()
        accelerations_filtered[:, step] = filtered_data[0:3]
        omega_filtered[:, step] = filtered_data[3:6]
        omega_dot_filtered[:, step] = filtered_data[6:9]
        controller_desired = angular_acceleration_controller.get_desired()
        positions_des[:, step] = controller_desired[10:13]
        velocities_des[:, step] = controller_desired[13:16]
        euler_des[:, step] = R.from_quat(
            controller_desired[0:4], scalar_first=True
        ).as_euler("ZXY")
        omega_des[:, step] = controller_desired[4:7]
        omega_dot_des[:, step] = controller_desired[7:10]
        accelerations_des[:, step] = controller_desired[10:13]
        # print(omega_dot_des[:, step])
        force_commanded[:, step], moment_commanded[:, step] = (
            angular_acceleration_controller.get_commanded_force_moment()
        )
        angle_error_vec[:, step] = (
            angular_acceleration_controller.angle_error_vec.copy()
        )

        # print(control_test)
        # print(np.linalg.norm(state_f[28:31]) / aircraft.mass)

    animate(
        states,
        controls,
        aircraft,
        hz,
        phi_data,
        fps=30,
        force_data=force_data,
        follow=follow,
        trail=trail,
        des_path=des_trail,
        des_path_data=positions_des,
        phi_data=phi_data,
        draw_phi_data=False,
        save_anim=False,
        file_name="slow_loop.mp4",
    )

    plt.show()

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("angle error vec")
    ax = axs[0]
    ax.set_title("x")
    ax.plot(t_steps, angle_error_vec[0, :])
    ax = axs[1]
    ax.set_title("y")
    ax.plot(t_steps, angle_error_vec[1, :])
    ax = axs[2]
    ax.set_title("z")
    ax.plot(t_steps, angle_error_vec[2, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("position vs desired")
    ax = axs[0]
    ax.set_title("x")
    ax.plot(t_steps, states[0, :])
    ax.plot(t_steps, positions_des[0, :])
    ax = axs[1]
    ax.set_title("y")
    ax.plot(t_steps, states[1, :])
    ax.plot(t_steps, positions_des[1, :])
    ax = axs[2]
    ax.set_title("z")
    ax.plot(t_steps, states[2, :])
    ax.plot(t_steps, positions_des[2, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("velocity inertial vs desired")
    ax = axs[0]
    ax.set_title("x")
    ax.plot(t_steps, velocities_inertial[0, :])
    ax.plot(t_steps, velocities_des[0, :])
    ax = axs[1]
    ax.set_title("y")
    ax.plot(t_steps, velocities_inertial[1, :])
    ax.plot(t_steps, velocities_des[1, :])
    ax = axs[2]
    ax.set_title("z")
    ax.plot(t_steps, velocities_inertial[2, :])
    ax.plot(t_steps, velocities_des[2, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("commanded forces")
    ax = axs[0]
    ax.set_title("x")
    ax.plot(t_steps, force_commanded[0, :])
    ax.plot(t_steps, force_achieved[0, :])
    ax = axs[1]
    ax.set_title("y")
    ax.plot(t_steps, force_commanded[1, :])
    ax.plot(t_steps, force_achieved[1, :])
    ax = axs[2]
    ax.set_title("z")
    ax.plot(t_steps, force_commanded[2, :])
    ax.plot(t_steps, force_achieved[2, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("acceleration inertial vs filtered")
    ax = axs[0]
    ax.set_title("x")
    ax.plot(t_steps, accelerations[0, :])
    ax.plot(t_steps, accelerations_filtered[0, :])
    ax.plot(t_steps, accelerations_des[0, :])
    ax = axs[1]
    ax.set_title("y")
    ax.plot(t_steps, accelerations[1, :])
    ax.plot(t_steps, accelerations_filtered[1, :])
    ax.plot(t_steps, accelerations_des[1, :])
    ax = axs[2]
    ax.set_title("z")
    ax.plot(t_steps, accelerations[2, :])
    ax.plot(t_steps, accelerations_filtered[2, :])
    ax.plot(t_steps, accelerations_des[2, :])

    # fig, axs = plt.subplots(3, 1)
    # fig.suptitle("orientation")
    # ax = axs[0]
    # ax.set_title("roll")
    # # ax.plot(t_steps, np.degrees(orientation[1, :]))
    # # ax.plot(t_steps, np.degrees(euler_des[1, :]))
    # ax.plot(t_steps, orientation[1, :])
    # ax.plot(t_steps, euler_des[1, :])
    # ax = axs[1]
    # ax.set_title("pitch")
    # # ax.plot(t_steps, np.degrees(orientation[2, :]))
    # # ax.plot(t_steps, np.degrees(euler_des[2, :]))
    # ax.plot(t_steps, orientation[2, :])
    # ax.plot(t_steps, euler_des[2, :])
    # ax = axs[2]
    # ax.set_title("yaw")
    # # ax.plot(t_steps, np.degrees(orientation[0, :]))
    # # ax.plot(t_steps, np.degrees(euler_des[0, :]))
    # ax.plot(t_steps, orientation[0, :])
    # ax.plot(t_steps, euler_des[0, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("orientation")
    ax = axs[0]
    ax.set_title("roll")
    ax.plot(t_steps, np.degrees(orientation[1, :]))
    ax.plot(t_steps, np.degrees(euler_des[1, :]))
    ax = axs[1]
    ax.set_title("pitch")
    ax.plot(t_steps, np.degrees(orientation[2, :]))
    ax.plot(t_steps, np.degrees(euler_des[2, :]))
    ax = axs[2]
    ax.set_title("yaw")
    ax.plot(t_steps, np.degrees(orientation[0, :]))
    ax.plot(t_steps, np.degrees(euler_des[0, :]))

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("angular velocity body vs filtered")
    ax = axs[0]
    ax.plot(t_steps, states[10, :])
    ax.plot(t_steps, omega_filtered[0, :])
    ax.plot(t_steps, omega_des[0, :])
    ax = axs[1]
    ax.plot(t_steps, states[11, :])
    ax.plot(t_steps, omega_filtered[1, :])
    ax.plot(t_steps, omega_des[1, :])
    ax = axs[2]
    ax.plot(t_steps, states[12, :])
    ax.plot(t_steps, omega_filtered[2, :])
    ax.plot(t_steps, omega_des[2, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("angular acceleration body vs filtered vs desired")
    ax = axs[0]
    ax.plot(t_steps, omega_dot[0, :])
    ax.plot(t_steps, omega_dot_filtered[0, :])
    ax.plot(t_steps, omega_dot_des[0, :])
    ax = axs[1]
    ax.plot(t_steps, omega_dot[1, :])
    ax.plot(t_steps, omega_dot_filtered[1, :])
    ax.plot(t_steps, omega_dot_des[1, :])
    ax = axs[2]
    ax.plot(t_steps, omega_dot[2, :])
    ax.plot(t_steps, omega_dot_filtered[2, :])
    ax.plot(t_steps, omega_dot_des[2, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("commanded moments")
    ax = axs[0]
    ax.plot(t_steps, moment_commanded[0, :])
    ax.plot(t_steps, moment_achieved[0, :])
    ax = axs[1]
    ax.plot(t_steps, moment_commanded[1, :])
    ax.plot(t_steps, moment_achieved[1, :])
    ax = axs[2]
    ax.plot(t_steps, moment_commanded[2, :])
    ax.plot(t_steps, moment_achieved[2, :])

    fig, axs = plt.subplots(4, 1)
    fig.suptitle("commands")
    ax = axs[0]
    ax.set_title("flap l")
    ax.plot(t_steps, np.degrees(states[13, :]))
    ax.plot(t_steps, np.degrees(controls_des[0, :]))
    ax = axs[1]
    ax.set_title("flap r")
    ax.plot(t_steps, np.degrees(states[14, :]))
    ax.plot(t_steps, np.degrees(controls_des[1, :]))
    ax = axs[2]
    ax.set_title("motor l")
    ax.plot(t_steps, np.degrees(states[15, :]))
    ax.plot(t_steps, np.degrees(controls_des[2, :]))
    ax = axs[3]
    ax.set_title("motor r")
    ax.plot(t_steps, np.degrees(states[16, :]))
    ax.plot(t_steps, np.degrees(controls_des[3, :]))
    plt.show()


if __name__ == "__main__":
    main()
