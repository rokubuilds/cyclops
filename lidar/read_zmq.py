# read_zmq.py
import struct
import zmq
import numpy as np
import open3d as o3d

REC_SIZE = 4 + 2 + 2 + 2 + 12  # azimuth f32 + 3*u16 + 3*f32  == 22 bytes

def parse_points(payload: bytes) -> np.ndarray | None:
    pts = []
    off = 0
    n = len(payload)

    # Heuristic: chew through full 22-byte records; ignore tail bytes
    while off + REC_SIZE <= n:
        az, tag1, tag2, tag3, x, y, z = struct.unpack_from("<fHHHfff", payload, off)
        # Optional sanity checks (azimuth 0..360, finite coordinates)
        if 0.0 <= az <= 360.0 and np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            pts.append((x, y, z))
        off += REC_SIZE

    if not pts:
        return None
    return np.asarray(pts, dtype=np.float32)

def main():
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect("tcp://192.168.1.249:5555")  # your point-cloud stream
    sock.setsockopt(zmq.SUBSCRIBE, b"")

    vis = o3d.visualization.Visualizer()
    vis.create_window("LiDAR Live Feed")
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    print(f"Listening on tcp://192.168.1.249:5555 …")
    while True:
        msg = sock.recv()
        pts = parse_points(msg)
        if pts is None or pts.size == 0:
            print(f"Skipped payload ({len(msg)} bytes) — no valid XYZ parsed.")
            continue

        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

if __name__ == "__main__":
    main()
