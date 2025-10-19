# imu_stick_figure.py
# Stick-figure visual driven by IMU quaternion over ZMQ (port 5556).
# Packet layout assumed: <III ffff fff fff>
#   seq(u32), stamp_sec(u32), stamp_nsec(u32),
#   quat(4f), ang_vel(3f), lin_acc(3f)
#
# If your quaternion is (w,x,y,z) instead of (x,y,z,w), toggle Q_IS_WXYZ below.

import math
import struct
import zmq
import numpy as np
import open3d as o3d

IMU_ENDPOINT = "tcp://192.168.1.249:5556"
Q_IS_WXYZ = False  # set True if your quat order is (w,x,y,z)

# ---------- helpers

def make_xy_grid(half=2.0, step=0.2, z=0.0, color=(0.7, 0.7, 0.7)):
    xs = np.arange(-half, half + 1e-9, step)
    ys = np.arange(-half, half + 1e-9, step)
    pts = []
    lines = []
    # verticals
    for i, x in enumerate(xs):
        pts += [[x, -half, z], [x, half, z]]
        lines.append([2 * i, 2 * i + 1])
    base = len(pts)
    # horizontals
    for j, y in enumerate(ys):
        pts += [[-half, y, z], [half, y, z]]
        lines.append([base + 2 * j, base + 2 * j + 1])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(pts))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return grid

def quat_to_R(qw, qx, qy, qz):
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) or 1.0
    w, x, y, z = qw/n, qx/n, qy/n, qz/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=np.float64)

def make_stick_figure():
    """
    Returns (lineset, rest_points, lines)
    rest_points are body-frame joints; we only rotate them each frame.
    """
    # Simple proportions (meters)
    hip = np.array([0, 0, 0.90])      # hip center height
    shoulder = np.array([0, 0, 1.45]) # shoulder height
    head_top = np.array([0, 0, 1.70])

    half_shoulder = 0.20
    half_hip = 0.15
    arm = 0.35
    leg = 0.45

    # joints (body frame)
    P = []
    # spine: hip -> shoulder -> head
    P += [hip, shoulder, head_top]
    # shoulders L/R
    P += [shoulder + np.array([ half_shoulder, 0, 0]),
          shoulder + np.array([-half_shoulder, 0, 0])]
    # elbows L/R
    P += [P[-2] + np.array([ arm, 0, 0]),
          P[-1] + np.array([-arm, 0, 0])]
    # hips L/R
    P += [hip + np.array([ half_hip, 0, 0]),
          hip + np.array([-half_hip, 0, 0])]
    # knees L/R
    P += [P[-2] + np.array([ 0, 0, -leg]),
          P[-1] + np.array([ 0, 0, -leg])]
    # feet L/R
    P += [P[-2] + np.array([ 0.10, 0, -0.05]),
          P[-1] + np.array([-0.10, 0, -0.05])]

    P = np.vstack(P)

    # connections (line segments by point index)
    idx = {
        "hip":0, "sho":1, "head":2, "sL":3, "sR":4, "eL":5, "eR":6,
        "hL":7, "hR":8, "kL":9, "kR":10, "fL":11, "fR":12
    }
    lines = [
        [idx["hip"], idx["sho"]], [idx["sho"], idx["head"]],         # spine & neck
        [idx["sho"], idx["sL"]],  [idx["sL"],  idx["eL"]],           # left arm
        [idx["sho"], idx["sR"]],  [idx["sR"],  idx["eR"]],           # right arm
        [idx["hip"], idx["hL"]],  [idx["hL"],  idx["kL"]], [idx["kL"], idx["fL"]], # left leg
        [idx["hip"], idx["hR"]],  [idx["hR"],  idx["kR"]], [idx["kR"], idx["fR"]], # right leg
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(P.copy())
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.8, 0.4], (len(lines), 1)))
    return ls, P, np.asarray(lines, dtype=np.int32)

# ---------- main

def main():
    # ZMQ
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(IMU_ENDPOINT)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    print(f"Listening IMU @ {IMU_ENDPOINT}")

    # Open3D scene
    vis = o3d.visualization.Visualizer()
    vis.create_window("IMU Stick Figure", width=960, height=720)
    grid = make_xy_grid(half=2.0, step=0.2, z=0.0)
    vis.add_geometry(grid)

    figure, rest_pts, line_idx = make_stick_figure()
    vis.add_geometry(figure)

    # Center camera nicely
    ctr = vis.get_view_control()
    bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-1.2,-1.2,-0.2]),
                                               np.array([ 1.2, 1.2, 2.0]))
    ctr.set_lookat([0,0,1.0])
    ctr.set_front([0,-1,0.1])
    ctr.set_up([0,0,1])
    ctr.set_zoom(0.8)

    while True:
        pkt = sock.recv()
        if len(pkt) < 52:
            continue

        # Unpack
        # seq, sec, nsec = struct.unpack_from("<III", pkt, 0)  # (unused)
        if Q_IS_WXYZ:
            qw, qx, qy, qz = struct.unpack_from("<ffff", pkt, 12)
        else:
            qx, qy, qz, qw = struct.unpack_from("<ffff", pkt, 12)

        # Build rotation from body-frame to world
        R = quat_to_R(qw, qx, qy, qz)

        # Rotate rest pose -> world pose (about origin)
        pts_world = (rest_pts @ R.T)

        # Slightly drop to stand on grid (optional)
        # pts_world[:,2] -= 0.9

        # Update LineSet points
        figure.points = o3d.utility.Vector3dVector(pts_world)

        vis.update_geometry(figure)
        vis.poll_events()
        vis.update_renderer()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
