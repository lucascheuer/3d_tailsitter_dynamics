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


follow_global = 0


class ForceLine:
    def __init__(
        self,
        base: np.array,
        force_vector: np.array,
        ax: Axes3D,
        force_scale=0.25,
        color="blue",
    ):
        self.base = base
        self.force_scale = force_scale
        self.force_vector = force_vector * self.force_scale
        self.base_line = np.zeros((2, 3))
        self.base_line[0, :] = self.base
        self.base_line[1, :] = self.force_vector + self.base
        line_data = self.base_line.copy()
        self.line = art3d.Line3D(
            line_data[:, 0], line_data[:, 1], line_data[:, 2], color=color
        )
        ax.add_line(self.line)

    # update the line pre-rotation and return it
    def update_base(self, force_vector: np.array):
        self.force_vector = force_vector * self.force_scale
        self.base_line[1, :] = self.force_vector + self.base
        return self.base_line

    # update the line object
    def update_line(self, line_data):
        self.line.set_data_3d(
            line_data[:, 0],
            line_data[:, 1],
            line_data[:, 2],
        )


class Aircraft:
    def __init__(
        self,
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
    ):
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.moment_of_inertia_inv = np.linalg.inv(self.moment_of_inertia)
        self.C_t = C_t  # propeller coefficient of thrust
        self.C_m = C_m  # propeller coefficient of moment
        self.max_omega = max_omega
        self.max_omega_dot = max_omega_dot
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
        self.max_elevon_dot = max_elevon_dot
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


def animate(
    states,
    aircraft: Aircraft,
    data_hz,
    fps=20,
    force_data=None,
    draw_force=False,
    force_scale=0.5,
    follow=0,
    trail=True,
    des_path_data=None,
    des_path=False,
    wps=False,
    wps_data=None,
    save_anim=False,
    file_name="tmp.mp4",
):
    global follow_global
    follow_global = follow
    frame_mult = int(data_hz / fps)
    elevon_chord = aircraft.chord * aircraft.elevon_percentage
    wing_chord = aircraft.chord - elevon_chord
    wing_front_back = aircraft.chord * 0.5
    wing_elevon_front = -wing_front_back + elevon_chord
    wing_tip = aircraft.wingspan * 0.5
    wing = np.zeros((4, 3))
    # front right
    wing[0, 0] = wing_front_back
    wing[0, 1] = wing_tip
    wing[0, 2] = 0.0
    # front left
    wing[1, 0] = wing_front_back
    wing[1, 1] = -wing_tip
    wing[1, 2] = 0.0
    # back left
    wing[2, 0] = wing_elevon_front
    wing[2, 1] = -wing_tip
    wing[2, 2] = 0.0
    # back right
    wing[3, 0] = wing_elevon_front
    wing[3, 1] = wing_tip
    wing[3, 2] = 0.0

    right_elevon = np.zeros((4, 3))
    # front right
    right_elevon[0, 0] = wing_elevon_front
    right_elevon[0, 1] = wing_tip
    right_elevon[0, 2] = 0.0
    # front left
    right_elevon[1, 0] = wing_elevon_front
    right_elevon[1, 1] = 0.0
    right_elevon[1, 2] = 0.0
    # back left
    right_elevon[2, 0] = -wing_front_back
    right_elevon[2, 1] = 0.0
    right_elevon[2, 2] = 0.0
    # back right
    right_elevon[3, 0] = -wing_front_back
    right_elevon[3, 1] = wing_tip
    right_elevon[3, 2] = 0.0

    left_elevon = np.zeros((4, 3))
    # front right
    left_elevon[0, 0] = wing_elevon_front
    left_elevon[0, 1] = 0.0
    left_elevon[0, 2] = 0.0
    # front left
    left_elevon[1, 0] = wing_elevon_front
    left_elevon[1, 1] = -wing_tip
    left_elevon[1, 2] = 0.0
    # back left
    left_elevon[2, 0] = -wing_front_back
    left_elevon[2, 1] = -wing_tip
    left_elevon[2, 2] = 0.0
    # back right
    left_elevon[3, 0] = -wing_front_back
    left_elevon[3, 1] = 0.0
    left_elevon[3, 2] = 0.0

    wing = wing * 1
    right_elevon = right_elevon * 1
    left_elevon = left_elevon * 1
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Close the hexagon by repeating the first vertex
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    poly = art3d.Poly3DCollection(
        [wing.copy(), right_elevon.copy(), left_elevon.copy()],
        alpha=0.7,
        color=["blue", "green", "red"],
    )

    ax.add_collection3d(poly)
    if trail:
        (line,) = ax.plot([0], [0], color="k")

    if des_path:
        des_path_data[:, 0] = des_path_data[:, 1]
        (des_path_line,) = ax.plot([0], [0], color="b")
        des_path_line.set_data_3d(
            des_path_data[0, :],
            -des_path_data[1, :],
            -des_path_data[2, :],
        )
        (des_point,) = ax.plot([], [], [], color="red", marker="o")
    if wps:
        wps_scatter = ax.scatter(
            wps_data[1, :], -wps_data[2, :], -wps_data[3, :], color="g"
        )

    # force arrows
    #     0:3         3:6            6:9         9:12       12:15     15:18           18:21              21:24            24:27           27:30           30:33           33:36
    # motor_thr_l, motor_thr_r, motor_drag_l, motor_drag_r, lift, force_gravity, elv_lift_reduc_l, elv_lift_reduc_r, rot_lift_wing, elv_thr_redir_l, elv_thr_redir_r, rot_lift_reduc_elv
    motor_thr_l = ForceLine(aircraft.p_l, force_data[0:3, 0], ax, force_scale)
    motor_thr_r = ForceLine(aircraft.p_r, force_data[3:6, 0], ax, force_scale)
    motor_drag_l = ForceLine(aircraft.p_l, force_data[6:9, 0], ax, force_scale, "green")
    motor_drag_r = ForceLine(
        aircraft.p_r, force_data[9:12, 0], ax, force_scale, "green"
    )
    lift = ForceLine(
        np.array([aircraft.chord * 0.25, 0, 0]),
        force_data[12:15, 0],
        ax,
        force_scale,
        "cyan",
    )
    force_gravity = ForceLine(
        np.array([aircraft.chord * 0.25 + aircraft.delta_r, 0, 0]),
        force_data[15:18, 0],
        ax,
        force_scale,
        "red",
    )
    elv_lift_reduc_l = ForceLine(
        [wing[0, 0] - wing_chord - elevon_chord * 0.5, -wing_tip * 0.5, 0],
        force_data[18:21, 0],
        ax,
        force_scale,
        "green",
    )
    elv_lift_reduc_r = ForceLine(
        [wing[0, 0] - wing_chord - elevon_chord * 0.5, wing_tip * 0.5, 0],
        force_data[21:24, 0],
        ax,
        force_scale,
        "green",
    )
    rot_lift_wing = ForceLine([0, 0, 0], force_data[24:27, 0], ax, force_scale)
    elv_thr_redir_l = ForceLine(
        [
            wing[0, 0] - wing_chord - elevon_chord * 0.5,
            -aircraft.wingspan * 0.25,
            0,
        ],
        force_data[27:30, 0],
        ax,
        force_scale,
    )
    elv_thr_redir_r = ForceLine(
        [
            wing[0, 0] - wing_chord - elevon_chord * 0.5,
            aircraft.wingspan * 0.25,
            0,
        ],
        force_data[30:33, 0],
        ax,
        force_scale,
    )
    rot_lift_reduc_elv = ForceLine(
        [0, 0, 0],
        force_data[33:36, 0],
        ax,
        force_scale,
    )
    x_min = states[0, 0] - 1
    x_max = states[0, 0] + 1
    y_min = states[1, 0] - 1
    y_max = states[1, 0] + 1
    z_min = states[2, 0] - 1
    z_max = states[2, 0] + 1
    # xy_min = min(min(states[0, :]), min(states[1, :]))
    # yx_max = max(max(states[0, :]), max(states[1, :]))
    # yx = max(abs(xy_min), abs(yx_max))
    # ax.set_xlim(-yx, yx)
    # ax.set_ylim(-yx, yx)
    # ax.set_zlim(min(states[2, :]), max(states[2, :]))
    ax.set_aspect("equalxy")
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)

    if follow_global == 2 and des_path:
        x_min = np.min(des_path_data[0, :]) - 1
        x_max = np.max(des_path_data[0, :]) + 1
        y_min = np.min(-des_path_data[1, :]) - 1
        y_max = np.max(-des_path_data[1, :]) + 1
        z_min = np.min(-des_path_data[2, :]) - 1
        z_max = np.max(-des_path_data[2, :]) + 1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    def change_camera(event=None):
        global follow_global

        follow_global += 1
        if follow_global > 2:
            follow_global = 0
        print(follow_global)

    # States x
    ################## Linear #############           ############# Rotational ##########################
    #   0      1      2      3      4       5      6       7       8       9      10       11       12
    # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, quat_w, quat_x, quat_y, quat_z, omega_x, omega_y, omega_z

    # Inputs u
    #    0       1         2          3
    # flap_l, flap_r, motor_w_l, motor_w_r
    def update(frame):
        # angle = np.pi * 2 / 50 * frame
        # elevon_angle = np.deg2rad(15) * np.sin(np.pi * 2 / 50 * frame)
        if trail:
            line.set_data_3d(
                states[0, : frame * frame_mult + 1 : 2],
                -states[1, : frame * frame_mult + 1 : 2],
                -states[2, : frame * frame_mult + 1 : 2],
            )
            # print(states[0, : frame * frame_mult + 1])
            # print()
        if des_path and frame * frame_mult < des_path_data.shape[1]:
            des_point.set_data(
                [des_path_data[0, frame * frame_mult]],
                [-des_path_data[1, frame * frame_mult]],
            )
            des_point.set_3d_properties([-des_path_data[2, frame * frame_mult]])
        left_elevon_rot = R.from_rotvec(
            states[13, frame * frame_mult] * np.array([0, 1, 0])
        )
        right_elevon_rot = R.from_rotvec(
            states[14, frame * frame_mult] * np.array([0, 1, 0])
        )

        # body_rot = R.from_rotvec(-angle * np.array([0, 0, 1]))
        # body_rot = R.from_euler("xyz", [np.deg2rad(75), 0, -angle])
        body_rot = R.from_quat(
            [
                states[6, frame * frame_mult],
                states[7, frame * frame_mult],
                states[8, frame * frame_mult],
                states[9, frame * frame_mult],
            ],
            scalar_first=True,
        )

        normal_frame = R.from_euler("xyz", [np.pi, 0, 0])
        new_left_elevon = left_elevon - [wing_elevon_front, 0, 0]
        new_right_elevon = right_elevon - [wing_elevon_front, 0, 0]
        new_left_elevon = left_elevon_rot.apply(new_left_elevon)
        new_right_elevon = right_elevon_rot.apply(new_right_elevon)
        new_left_elevon += [wing_elevon_front, 0, 0]
        new_right_elevon += [wing_elevon_front, 0, 0]
        mtr_thr_l_data = motor_thr_l.update_base(
            force_data[0:3, frame * frame_mult] * force_scale
        )
        mtr_thr_r_data = motor_thr_r.update_base(
            force_data[3:6, frame * frame_mult] * force_scale
        )
        mtr_drg_l_data = motor_drag_l.update_base(
            force_data[6:9, frame * frame_mult] * force_scale
        )
        mtr_drg_r_data = motor_drag_r.update_base(
            force_data[9:12, frame * frame_mult] * force_scale
        )
        lift_data = lift.update_base(
            force_data[12:15, frame * frame_mult] * force_scale
        )
        force_gravity_data = force_gravity.update_base(
            force_data[15:18, frame * frame_mult] * force_scale
        )
        elv_lift_reduc_l_data = elv_lift_reduc_l.update_base(
            force_data[18:21, frame * frame_mult] * force_scale
        )
        elv_lift_reduc_r_data = elv_lift_reduc_r.update_base(
            force_data[21:24, frame * frame_mult] * force_scale
        )
        rot_lift_wing_data = rot_lift_wing.update_base(
            force_data[24:27, frame * frame_mult] * force_scale
        )
        elv_thr_redir_l_data = elv_thr_redir_l.update_base(
            force_data[27:30, frame * frame_mult] * force_scale
        )
        elv_thr_redir_r_data = elv_thr_redir_r.update_base(
            force_data[30:33, frame * frame_mult] * force_scale
        )
        rot_lift_reduc_elv_data = rot_lift_reduc_elv.update_base(
            force_data[33:36, frame * frame_mult] * force_scale
        )
        full_body = np.concatenate(
            (
                wing,
                new_right_elevon,
                new_left_elevon,
                mtr_thr_l_data,
                mtr_thr_r_data,
                mtr_drg_l_data,
                mtr_drg_r_data,
                lift_data,
                force_gravity_data,
                elv_lift_reduc_l_data,
                elv_lift_reduc_r_data,
                rot_lift_wing_data,
                elv_thr_redir_l_data,
                elv_thr_redir_r_data,
                rot_lift_reduc_elv_data,
            ),
            axis=0,
        ) + np.array([-aircraft.chord * 0.25, 0, 0])

        # rotate to correct body rot
        full_body = body_rot.apply(full_body)
        full_body = normal_frame.apply(full_body)
        # move to proper location
        full_body += np.array(
            [
                states[0, frame * frame_mult],
                -states[1, frame * frame_mult],
                -states[2, frame * frame_mult],
            ]
        )

        new_wing = full_body[:4, :]
        new_right_elevon = full_body[4:8, :]
        new_left_elevon = full_body[8:12, :]
        mtr_thr_l_data = full_body[12:14, :]
        mtr_thr_r_data = full_body[14:16, :]
        mtr_drg_l_data = full_body[16:18, :]
        mtr_drg_r_data = full_body[18:20, :]
        lift_data = full_body[20:22, :]
        force_gravity_data = full_body[22:24, :]
        # print(np.linalg.norm(force_gravity.force_vector / aircraft.mass))
        elv_lift_reduc_l_data = full_body[24:26, :]
        elv_lift_reduc_r_data = full_body[26:28, :]
        rot_lift_wing_data = full_body[28:30, :]
        elv_thr_redir_l_data = full_body[30:32, :]
        elv_thr_redir_r_data = full_body[32:34, :]
        rot_lift_reduc_elv_data = full_body[34:36, :]
        poly.set_verts([new_wing, new_right_elevon, new_left_elevon])
        motor_thr_l.update_line(mtr_thr_l_data)
        motor_thr_r.update_line(mtr_thr_r_data)
        motor_drag_l.update_line(mtr_drg_l_data)
        motor_drag_r.update_line(mtr_drg_r_data)
        lift.update_line(lift_data)
        force_gravity.update_line(force_gravity_data)
        elv_lift_reduc_l.update_line(elv_lift_reduc_l_data)
        elv_lift_reduc_r.update_line(elv_lift_reduc_r_data)
        rot_lift_wing.update_line(rot_lift_wing_data)
        elv_thr_redir_l.update_line(elv_thr_redir_l_data)
        elv_thr_redir_r.update_line(elv_thr_redir_r_data)
        rot_lift_reduc_elv.update_line(rot_lift_reduc_elv_data)
        # Setting the size of the view
        if follow_global == 0:
            if frame == 0:
                camera_size = 5
                ax.set_xlim(
                    states[0, frame * frame_mult] - camera_size,
                    states[0, frame * frame_mult] + camera_size,
                )
                ax.set_ylim(
                    -states[1, frame * frame_mult] - camera_size,
                    -states[1, frame * frame_mult] + camera_size,
                )
                ax.set_zlim(
                    -states[2, frame * frame_mult] - camera_size,
                    -states[2, frame * frame_mult] + camera_size,
                )
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            z_min, z_max = ax.get_zlim()
            x_min = min(x_min, states[0, frame * frame_mult] - 1)
            x_max = max(x_max, states[0, frame * frame_mult] + 1)
            y_min = min(y_min, -states[1, frame * frame_mult] - 1)
            y_max = max(y_max, -states[1, frame * frame_mult] + 1)
            z_min = min(z_min, -states[2, frame * frame_mult] - 1)
            z_max = max(z_max, -states[2, frame * frame_mult] + 1)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
        elif follow_global == 1 or not des_path and follow_global == 2:
            ax.set_xlim(
                states[0, frame * frame_mult] - 1, states[0, frame * frame_mult] + 1
            )
            ax.set_ylim(
                -states[1, frame * frame_mult] - 1, -states[1, frame * frame_mult] + 1
            )
            ax.set_zlim(
                -states[2, frame * frame_mult] - 1, -states[2, frame * frame_mult] + 1
            )
        elif follow_global == 2 and des_path:
            x_min = np.min(des_path_data[0, :]) - 1
            x_max = np.max(des_path_data[0, :]) + 1
            y_min = np.min(-des_path_data[1, :]) - 1
            y_max = np.max(-des_path_data[1, :]) + 1
            z_min = np.min(-des_path_data[2, :]) - 1
            z_max = np.max(-des_path_data[2, :]) + 1
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

        ax.set_aspect("equal")

    # phi_y = phi_rot.inv().apply([0, 1, 0])
    # b_y = body_rot.apply([0, 1, 0])
    # dot = np.dot(b_y, phi_y)
    # print(
    #     frame, phi[frame * frame_mult], theta[frame * frame_mult], phi_y, b_y, dot
    # )
    # return poly

    last_frame = int(states.shape[1] / frame_mult) - 1
    print("last frame:", last_frame)
    ani = Player(
        fig,
        update,
        frames=range(last_frame),
        interval=1 / fps * 1000,
        blit=False,
        maxi=last_frame,
        save_count=last_frame,
    )
    ani.button_camera_switch.on_clicked(change_camera)

    plt.show()
    if save_anim:
        ani.save(
            filename=file_name,
            writer="ffmpeg",
            fps=fps,
            progress_callback=lambda i, n: print(f"Saving frame {i}/{n}"),
        )


def find_data_animate(
    state_file,
    force_file,
    controls_file,
    aircraft_model_params_file,
    run_settings_file,
    waypoint_file,
):
    with open(state_file, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        states = []
        next(csv_reader)
        for row in csv_reader:
            float_row = [float(item) for item in row]
            states.append(float_row)
            # print(len(float_row))
        states = np.array(states).T
        states = states[1:, :]
    with open(force_file, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        forces = []
        # next(csv_reader)
        for row in csv_reader:
            float_row = [float(item) for item in row]
            forces.append(float_row)
        forces = np.array(forces).T
        forces = forces[1:, :]
    with open(controls_file, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        des_pos = []
        next(csv_reader)
        for row in csv_reader:
            float_row = [float(item) for item in row[0:4]]
            des_pos.append(float_row)
        des_pos = np.array(des_pos).T
        des_pos = des_pos[1:, :]
        print(des_pos.shape)
    with open(aircraft_model_params_file, "rb") as f:
        config_data = tomllib.load(f)
        mass = config_data["mass"]
        wingspan = config_data["wingspan"]
        chord = config_data["chord"]
        depth = config_data["depth"]
        moment_of_inertia = np.eye(3)
        moment_of_inertia[0, 0] = 1 / 12 * mass * (wingspan**2 + chord**2)
        moment_of_inertia[1, 1] = 1 / 12 * mass * (depth**2 + chord**2)
        moment_of_inertia[2, 2] = 1 / 12 * mass * (wingspan**2 + depth**2)
        C_t = config_data["prop_thrust_coeff"]
        C_m = config_data["prop_moment_coeff"]
        S = wingspan * chord  # surface area
        prop_diameter = config_data["propeller_diameter"]
        S_p = np.pi * prop_diameter**2  # each prop is span in diameter
        C_d_naught = config_data["minimum_drag_coeff"]
        C_y_naught = config_data["maximum_drag_coeff"]
        C_l_p = config_data["roll_rate_drag_coeff"]
        C_m_q = config_data["pitch_rate_drag_coeff"]
        C_n_r = config_data["yaw_rate_drag_coeff"]
        elevon_effectiveness_linear = np.array(
            config_data["linear_elevon_effectiveness"]
        )
        elevon_effectiveness_rotational = np.array(
            config_data["rotational_elevon_effectiveness"]
        )
        elevon_percentage = config_data["elevon_percentage"]
        max_elevon_angle = config_data["max_elevon_angle"]
        max_elevon_dot = config_data["max_elevon_angle_dot"]  # rad/s
        max_omega = config_data["prop_max_omega"]
        max_omega_dot = config_data[
            "prop_max_omega_dot"
        ]  # max_omega / 0.025  # rad/s/s
        delta_r = config_data["delta_r"]
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
    with open(run_settings_file, "rb") as f:
        config_data = tomllib.load(f)
        freq = 1 / config_data["time_step"]
        print(freq)
    with open(waypoint_file, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        wps = []
        next(csv_reader)
        for row in csv_reader:
            float_row = [float(item) for item in row]
            wps.append(float_row)

        wps = np.array(wps).T
    animate(
        states,
        aircraft,
        freq,
        force_data=forces,
        follow=0,
        fps=30,
        des_path_data=des_pos,
        des_path=True,
        wps=True,
        wps_data=wps,
        save_anim=False,
    )
