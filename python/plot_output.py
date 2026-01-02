import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from player import Player
import csv
import tomllib

# from scipy.spatial.transform import RigidTransform as Tf
from scipy.spatial.transform import Rotation as R


def plot_output(states, states_dot, control_data):

    # states[7, :]  # w
    # states[8, :]  # x
    # states[9, :]  # y
    # states[10, :]  # z
    # t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z,omega_x,omega_y,omega_z,elevon_left,elevon_right,motor_omega_left,motor_omega_right
    np.set_printoptions(suppress=True, precision=6)

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("position")
    ax = axs[0]
    ax.set_title("x")
    ax.plot(states[0, :], states[1, :])
    ax.plot(control_data[0, :], control_data[1, :])

    ax = axs[1]
    ax.set_title("y")
    ax.plot(states[0, :], states[2, :])
    ax.plot(control_data[0, :], control_data[2, :])

    ax = axs[2]
    ax.set_title("z")
    ax.plot(states[0, :], states[3, :])
    ax.plot(control_data[0, :], control_data[3, :])
    r = np.sqrt(states[1, -1] ** 2 + states[2, -1] ** 2)
    z_pos = states[3, -1]

    quats = states[7:11, :].T
    quats_xyzw = quats[:, [1, 2, 3, 0]]
    rotations = R.from_quat(quats_xyzw)
    euler_angles_deg = rotations.as_euler("ZXY", degrees=True).T

    quats_des = control_data[7:11, :].T
    quats_xyzw = quats_des[:, [1, 2, 3, 0]]
    rotations_des = R.from_quat(quats_xyzw)
    euler_angles_deg_des = rotations_des.as_euler("ZXY", degrees=True).T

    velocity_body = states[4:7, :]
    velocity_inertial_des = control_data[4:7, :]
    velocity_inertial = rotations.apply(velocity_body.T).T
    velocity_body_des = rotations[:-1].inv().apply(velocity_inertial_des.T).T
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("velocity inertial")
    ax = axs[0]
    ax.set_title("x")
    ax.plot(states[0, :], velocity_inertial[0, :])
    ax.plot(control_data[0, :], velocity_inertial_des[0, :])
    ax = axs[1]
    ax.set_title("y")
    ax.plot(states[0, :], velocity_inertial[1, :])
    ax.plot(control_data[0, :], velocity_inertial_des[1, :])
    ax = axs[2]
    ax.set_title("z")
    ax.plot(states[0, :], velocity_inertial[2, :])
    ax.plot(control_data[0, :], velocity_inertial_des[2, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("velocity body")
    ax = axs[0]
    ax.set_title("x")
    ax.plot(states[0, :], velocity_body[0, :])
    ax.plot(control_data[0, :], velocity_body_des[0, :])
    ax = axs[1]
    ax.set_title("y")
    ax.plot(states[0, :], velocity_body[1, :])
    ax.plot(control_data[0, :], velocity_body_des[1, :])
    ax = axs[2]
    ax.set_title("z")
    ax.plot(states[0, :], velocity_body[2, :])
    ax.plot(control_data[0, :], velocity_body_des[2, :])

    fig, axs = plt.subplots(4, 1)
    fig.suptitle("quats")
    ax = axs[0]
    ax.set_title("w")
    ax.plot(states[0, :], states[7, :])
    ax.plot(control_data[0, :], control_data[7, :])
    ax = axs[1]
    ax.set_title("x")
    ax.plot(states[0, :], states[8, :])
    ax.plot(control_data[0, :], control_data[8, :])
    ax = axs[2]
    ax.set_title("y")
    ax.plot(states[0, :], states[9, :])
    ax.plot(control_data[0, :], control_data[9, :])
    ax = axs[3]
    ax.set_title("z")
    ax.plot(states[0, :], states[10, :])
    ax.plot(control_data[0, :], control_data[10, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("orientation")
    ax = axs[0]
    ax.set_title("roll")
    ax.plot(states[0, :], euler_angles_deg[1, :])
    ax.plot(control_data[0, :], euler_angles_deg_des[1, :])
    ax = axs[1]
    ax.set_title("pitch")
    ax.plot(states[0, :], euler_angles_deg[2, :])
    ax.plot(control_data[0, :], euler_angles_deg_des[2, :])
    ax = axs[2]
    ax.set_title("yaw")
    ax.plot(states[0, :], euler_angles_deg[0, :])
    ax.plot(control_data[0, :], euler_angles_deg_des[0, :])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("angular velocity body")
    ax = axs[0]
    ax.set_title("x")
    ax.plot(states[0, :], states[11, :])
    ax.plot(control_data[0, :], control_data[11, :])
    ax = axs[1]
    ax.set_title("y")
    ax.plot(states[0, :], states[12, :])
    ax.plot(control_data[0, :], control_data[12, :])
    ax = axs[2]
    ax.set_title("z")
    ax.plot(states[0, :], states[13, :])
    ax.plot(control_data[0, :], control_data[13, :])

    # fig, axs = plt.subplots(3, 1)
    # fig.suptitle("commanded moments")
    # ax = axs[0]
    # ax.plot(states[0, :], moment_commanded[0, :])
    # ax.plot(states[0, :], moment_achieved[0, :])
    # ax = axs[1]
    # ax.plot(states[0, :], moment_commanded[1, :])
    # ax.plot(states[0, :], moment_achieved[1, :])
    # ax = axs[2]
    # ax.plot(states[0, :], moment_commanded[2, :])
    # ax.plot(states[0, :], moment_achieved[2, :])

    fig, axs = plt.subplots(4, 1)
    fig.suptitle("commands")
    ax = axs[0]
    ax.set_title("flap l")
    ax.plot(states[0, :], np.degrees(states[14, :]))
    ax.plot(control_data[0, :], np.degrees(control_data[14, :]))
    ax = axs[1]
    ax.set_title("flap r")
    ax.plot(states[0, :], np.degrees(states[15, :]))
    ax.plot(control_data[0, :], np.degrees(control_data[15, :]))
    ax = axs[2]
    ax.set_title("motor l")
    ax.plot(states[0, :], states[16, :])
    ax.plot(control_data[0, :], control_data[16, :])
    ax = axs[3]
    ax.set_title("motor r")
    ax.plot(states[0, :], states[17, :])
    ax.plot(control_data[0, :], control_data[17, :])

    print("pos_x = 0.0")
    print("pos_y = " + str(r))
    print("pos_z = " + str(z_pos))
    print("vel_x_body = " + str(velocity_body[0, -1]))
    print("vel_y_body = " + str(velocity_body[1, -1]))
    print("vel_z_body = " + str(velocity_body[2, -1]))
    print("roll = " + str(euler_angles_deg[1, -1]))
    print("pitch = " + str(euler_angles_deg[2, -1]))
    print("yaw = 0.0")
    print("omega_x = " + str(states[11, -1]))
    print("omega_y = " + str(states[12, -1]))
    print("omega_z = " + str(states[13, -1]))
    print("elevon_left = " + str(np.degrees(control_data[14, -1])))
    print("elevon_right = " + str(np.degrees(control_data[15, -1])))
    print("motor_omega_left = " + str(control_data[16, -1]))
    print("motor_omega_right = " + str(control_data[17, -1]))
    print()
    plt.show()


def find_data_plot(state_file, state_dot_file, control_file):
    with open(state_file, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        states = []
        next(csv_reader)
        for row in csv_reader:
            float_row = [float(item) for item in row]
            states.append(float_row)

        states = np.array(states).T
    with open(state_dot_file, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        states_dot = []
        next(csv_reader)
        for row in csv_reader:
            float_row = [float(item) for item in row]
            states_dot.append(float_row)

        states_dot = np.array(states_dot).T
    with open(control_file, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        control_data = []
        next(csv_reader)
        for row in csv_reader:
            float_row = [float(item) for item in row]
            control_data.append(float_row)

        control_data = np.array(control_data).T

    plot_output(states, states_dot, control_data)
