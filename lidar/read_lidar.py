import zmq
import numpy as np
import open3d as o3d  # pip install open3d

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://192.168.123.161:5555")  # replace IP with your LiDAR IP
socket.setsockopt_string(zmq.SUBSCRIBE, "")

vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

while True:
    msg = socket.recv()
    points = np.frombuffer(msg, dtype=np.float32).reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
