import numpy as np
import csv

# Define the data


def generate_circle(
    t_start,
    t_len,
    circle_frequency,
    diameter,
    x_offset=0,
    y_offset=0,
    z_offset=0,
    spiral_speed=0,
    yaw_offset_deg=0,
):
    circle_amplitude = diameter / 2
    yaw_offset = yaw_offset_deg / 180 * np.pi
    t_end = t_start + t_len
    t_step = 0.001
    times = np.arange(t_start, t_end, t_step)
    trig_times = times - times[0]
    pos_x = circle_amplitude * np.sin(np.pi * 2.0 * circle_frequency * trig_times)
    pos_x -= pos_x[0]
    pos_x += x_offset
    vel_x = (
        circle_amplitude
        * np.pi
        * 2.0
        * circle_frequency
        * np.cos(np.pi * 2.0 * circle_frequency * trig_times)
    )
    acc_x = (
        -circle_amplitude
        * np.pi**2
        * 2.0**2
        * circle_frequency**2
        * np.sin(np.pi * 2.0 * circle_frequency * trig_times)
    )
    jerk_x = (
        -circle_amplitude
        * np.pi**3
        * 2.0**3
        * circle_frequency**3
        * np.cos(np.pi * 2.0 * circle_frequency * trig_times)
    )
    snap_x = (
        circle_amplitude
        * np.pi**4
        * 2.0**4
        * circle_frequency**4
        * np.sin(np.pi * 2.0 * circle_frequency * trig_times)
    )

    pos_y = circle_amplitude * np.cos(np.pi * 2.0 * circle_frequency * trig_times)
    pos_y -= pos_y[0]
    pos_y += y_offset
    vel_y = (
        -circle_amplitude
        * np.pi
        * 2.0
        * circle_frequency
        * np.sin(np.pi * 2.0 * circle_frequency * trig_times)
    )
    acc_y = (
        -circle_amplitude
        * np.pi**2
        * 2.0**2
        * circle_frequency**2
        * np.cos(np.pi * 2.0 * circle_frequency * trig_times)
    )
    jerk_y = (
        circle_amplitude
        * np.pi**3
        * 2.0**3
        * circle_frequency**3
        * np.sin(np.pi * 2.0 * circle_frequency * trig_times)
    )
    snap_y = (
        circle_amplitude
        * np.pi**4
        * 2.0**4
        * circle_frequency**4
        * np.cos(np.pi * 2.0 * circle_frequency * trig_times)
    )

    pos_z = trig_times * spiral_speed
    pos_z += z_offset
    vel_z = np.ones(times.shape) * spiral_speed
    acc_z = np.zeros(times.shape)
    jerk_z = np.zeros(times.shape)
    snap_z = np.zeros(times.shape)
    yaw = -trig_times * np.pi * 2.0 * circle_frequency + yaw_offset
    yaw_dot = np.ones(times.shape) * -np.pi * 2.0 * circle_frequency
    data = np.column_stack(
        (
            times,
            pos_x,
            pos_y,
            pos_z,
            vel_x,
            vel_y,
            vel_z,
            acc_x,
            acc_y,
            acc_z,
            jerk_x,
            jerk_y,
            jerk_z,
            yaw,
            yaw_dot,
            snap_x,
            snap_y,
            snap_z,
        )
    )
    return data


if __name__ == "__main__":
    header_text = "times,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,jerk_x,jerk_y,jerk_z,yaw,yaw_dot"

    diameter = 4.0
    t_start = 0.0
    t_len = 8.0
    circle_frequency = 1.0 / t_len
    x_offset = 2
    y_offset = -2
    yaw_offset = 0
    spiral_speed = 0.0

    data = generate_circle(
        t_start,
        t_len,
        circle_frequency,
        diameter,
        x_offset,
        y_offset,
        spiral_speed,
        yaw_offset,
    )
    print("pos", data[0, 1:4])
    print("vel", data[0, 4:7])
    print("yaw, yaw_rate", data[0, -2:])
    np.savetxt(
        "trajectory_files/trajectory_circle.csv",
        data,
        fmt="%.5f",
        delimiter=",",
        header=header_text,
        comments="",
    )
