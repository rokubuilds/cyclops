# imu_attitude_open3d.py
import struct, numpy as np, zmq, open3d as o3d

IMU_ENDPOINT = "tcp://192.168.1.249:5556"

def quat_xyzw_to_o3d(qx,qy,qz,qw):
    # Open3D expects [w, x, y, z]
    return np.array([qw, qx, qy, qz], dtype=np.float64)

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect(IMU_ENDPOINT)
sock.setsockopt(zmq.SUBSCRIBE, b"")

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="IMU Attitude")
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
vis.add_geometry(frame)

# We’ll “undo” the previous rotation each frame so we can apply an absolute one.
R_prev = np.eye(3)

while True:
    data = sock.recv()
    if len(data) != 52:
        continue

    qx,qy,qz,qw = struct.unpack_from("<ffff", data, 12)
    quat_wxyz = quat_xyzw_to_o3d(qx,qy,qz,qw)
    R = o3d.geometry.get_rotation_matrix_from_quaternion(quat_wxyz)

    # Remove previous rotation, then apply current (absolute orientation)
    frame.rotate(R_prev.T, center=(0,0,0))
    frame.rotate(R,       center=(0,0,0))
    R_prev = R

    vis.update_geometry(frame)
    vis.poll_events()
    vis.update_renderer()
