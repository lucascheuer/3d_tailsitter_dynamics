import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from player import Player

# from scipy.spatial.transform import RigidTransform as Tf
from scipy.spatial.transform import Rotation as R

from global_dynamics import Aircraft

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


# paused = False


def animate(
    states,
    controls,
    aircraft: Aircraft,
    data_hz,
    fps=20,
    force_data=None,
    force_scale=0.5,
    follow=0,
    trail=False,
    des_path_data=None,
    des_path=True,
    phi_data=None,
    draw_phi_data=False,
    save_anim=False,
    file_name="tmp.mp4",
):
    global follow_global
    follow_global = follow
    frame_mult = int(data_hz / fps)
    print(frame_mult)
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
    if draw_phi_data:
        body_y = ForceLine([0, 0, 0], np.array([0, 1, 0]), ax, 1, "red")
        yaw_phi_y = ForceLine([0, 0, 0], np.array([0, 1, 0]), ax, 1, "green")

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
        if des_path:
            des_point.set_data(
                [des_path_data[0, frame * frame_mult]],
                [-des_path_data[1, frame * frame_mult]],
            )
            des_point.set_3d_properties([-des_path_data[2, frame * frame_mult]])
        left_elevon_rot = R.from_rotvec(
            controls[0, frame * frame_mult] * np.array([0, 1, 0])
        )
        right_elevon_rot = R.from_rotvec(
            controls[1, frame * frame_mult] * np.array([0, 1, 0])
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
        if draw_phi_data:
            body_y_data = body_y.update_base(np.array([0, 5, 0]))
            # move to quarter chord
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
                    body_y_data,
                ),
                axis=0,
            ) + np.array([-aircraft.chord * 0.25, 0, 0])
        else:
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
        if draw_phi_data:
            body_y_data = full_body[36:39, :]
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
        if draw_phi_data:
            body_y.update_line(body_y_data)

            phi_rot = phi_data[frame * frame_mult]

            yaw_phi_y_data = np.copy(yaw_phi_y.update_base(np.array([0, 5, 0])))
            yaw_phi_y_data += np.array([-aircraft.chord * 0.25, 0, 0])

            # rotate to correct body rot
            yaw_phi_y_data = phi_rot.apply(yaw_phi_y_data)
            yaw_phi_y_data = normal_frame.apply(yaw_phi_y_data)
            # move to proper location
            yaw_phi_y_data += np.array(
                [
                    states[0, frame * frame_mult],
                    -states[1, frame * frame_mult],
                    -states[2, frame * frame_mult],
                ]
            )
            yaw_phi_y.update_line(yaw_phi_y_data)
        # Setting the size of the view
        if follow_global == 0:
            if frame == 0:
                ax.set_xlim(
                    states[0, frame * frame_mult] - 1, states[0, frame * frame_mult] + 1
                )
                ax.set_ylim(
                    -states[1, frame * frame_mult] - 1,
                    -states[1, frame * frame_mult] + 1,
                )
                ax.set_zlim(
                    -states[2, frame * frame_mult] - 1,
                    -states[2, frame * frame_mult] + 1,
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


def animate_simple(states, controls, aircraft: Aircraft):
    wing = np.zeros((4, 3))
    # front right
    wing[0, 0] = aircraft.chord / 2
    wing[0, 1] = aircraft.wingspan / 2
    wing[0, 2] = 0.0
    # front left
    wing[1, 0] = aircraft.chord / 2
    wing[1, 1] = -aircraft.wingspan / 2
    wing[1, 2] = 0.0
    # back left
    wing[2, 0] = 0.0
    wing[2, 1] = -aircraft.wingspan / 2
    wing[2, 2] = 0.0
    # back right
    wing[3, 0] = 0.0
    wing[3, 1] = aircraft.wingspan / 2
    wing[3, 2] = 0.0

    right_elevon = np.zeros((4, 3))
    # front right
    right_elevon[0, 0] = 0.0
    right_elevon[0, 1] = aircraft.wingspan / 2
    right_elevon[0, 2] = 0.0
    # front left
    right_elevon[1, 0] = 0.0
    right_elevon[1, 1] = 0.0
    right_elevon[1, 2] = 0.0
    # back left
    right_elevon[2, 0] = -aircraft.chord / 2
    right_elevon[2, 1] = 0.0
    right_elevon[2, 2] = 0.0
    # back right
    right_elevon[3, 0] = -aircraft.chord / 2
    right_elevon[3, 1] = aircraft.wingspan / 2
    right_elevon[3, 2] = 0.0

    left_elevon = np.zeros((4, 3))
    # front right
    left_elevon[0, 0] = 0.0
    left_elevon[0, 1] = 0.0
    left_elevon[0, 2] = 0.0
    # front left
    left_elevon[1, 0] = 0.0
    left_elevon[1, 1] = -aircraft.wingspan / 2
    left_elevon[1, 2] = 0.0
    # back left
    left_elevon[2, 0] = -aircraft.chord / 2
    left_elevon[2, 1] = -aircraft.wingspan / 2
    left_elevon[2, 2] = 0.0
    # back right
    left_elevon[3, 0] = -aircraft.chord / 2
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

    print(np.concatenate((wing, right_elevon, left_elevon), axis=0).shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    print(ax)
    poly = art3d.Poly3DCollection(
        [wing.copy(), right_elevon.copy(), left_elevon.copy()],
        alpha=0.7,
        color=["red", "blue", "green"],
    )

    ax.add_collection3d(poly)

    print(
        min(states[0, :]),
        max(states[0, :]),
        min(states[1, :]),
        max(states[1, :]),
        min(states[2, :]),
        max(states[2, :]),
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
        left_elevon_rot = R.from_rotvec(
            controls[0, frame * frame_mult] * np.array([0, 1, 0])
        )
        right_elevon_rot = R.from_rotvec(
            controls[1, frame * frame_mult] * np.array([0, 1, 0])
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

        new_left_elevon = left_elevon_rot.apply(left_elevon)
        new_right_elevon = right_elevon_rot.apply(right_elevon)
        # move to quarter chord
        full_body = np.concatenate(
            (wing, new_right_elevon, new_left_elevon),
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
        poly.set_verts([new_wing, new_right_elevon, new_left_elevon])

        # Setting the size of the view
        # ax.set_xlim(states[0, frame * frame_mult] - 1, states[0, frame * frame_mult] + 1)
        # ax.set_ylim(-states[1, frame * frame_mult] - 1, -states[1, frame * frame_mult] + 1)
        # ax.set_zlim(-states[2, frame * frame_mult] - 1, -states[2, frame * frame_mult] + 1)
        if frame == 0:
            ax.set_xlim(
                states[0, frame * frame_mult] - 1, states[0, frame * frame_mult] + 1
            )
            ax.set_ylim(
                -states[1, frame * frame_mult] - 1, -states[1, frame * frame_mult] + 1
            )
            ax.set_zlim(
                -states[2, frame * frame_mult] - 1, -states[2, frame * frame_mult] + 1
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
        ax.set_aspect("equal")
        # return poly

    plt.show()


if __name__ == "__main__":
    animate()
