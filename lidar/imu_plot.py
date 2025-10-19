#!/usr/bin/env python3
import argparse
import struct
import time
from collections import deque

import numpy as np
import zmq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

IMU_ENDPOINT = "tcp://192.168.1.249:5556"

# Set True if your quaternion order on the wire is (w,x,y,z)
Q_IS_WXYZ = False

PACK_FMT = "<IIIffffffffff"  # 3*uint32 + 10*float32 = 12 + 40 = 52 bytes
PACK_LEN = 52

def parse(packet):
    """Return dict with seq, t (seconds), quat=(w,x,y,z), gyro=(x,y,z), accel=(x,y,z)."""
    if len(packet) < PACK_LEN:
        return None
    seq, sec, nsec, *floats = struct.unpack(PACK_FMT, packet[:PACK_LEN])
    if Q_IS_WXYZ:
        qw, qx, qy, qz = floats[0:4]
    else:
        qx, qy, qz, qw = floats[0:4]
    gx, gy, gz = floats[4:7]
    ax, ay, az = floats[7:10]
    # use device time if nonzero, else wall time
    t = sec + nsec * 1e-9
    if t == 0:
        t = time.time()
    return dict(
        seq=seq,
        t=t,
        quat=(qw, qx, qy, qz),
        gyro=(gx, gy, gz),
        accel=(ax, ay, az),
    )

def main():
    ap = argparse.ArgumentParser(description="Live IMU plots from ZMQ")
    ap.add_argument("--endpoint", default=IMU_ENDPOINT, help="ZMQ SUB endpoint")
    ap.add_argument("--window", type=float, default=20.0, help="seconds shown in plot window")
    ap.add_argument("--max-samples", type=int, default=20000, help="ring buffer size")
    ap.add_argument("--save", action="store_true", help="save imu_log.csv on exit")
    args = ap.parse_args()

    # --- ZMQ
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.RCVTIMEO = 100  # ms (non-blocking-ish)

    # --- buffers
    maxlen = args.max_samples
    t0 = None

    t_rel = deque(maxlen=maxlen)
    q_w = deque(maxlen=maxlen); q_x = deque(maxlen=maxlen); q_y = deque(maxlen=maxlen); q_z = deque(maxlen=maxlen)
    g_x = deque(maxlen=maxlen); g_y = deque(maxlen=maxlen); g_z = deque(maxlen=maxlen)
    a_x = deque(maxlen=maxlen); a_y = deque(maxlen=maxlen); a_z = deque(maxlen=maxlen)

    # --- plotting setup
    plt.style.use("default")
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    ax1.set_title("Angular velocity (rad/s)")
    l_gx, = ax1.plot([], [], label="gx")
    l_gy, = ax1.plot([], [], label="gy")
    l_gz, = ax1.plot([], [], label="gz")
    ax1.legend(loc="upper right"); ax1.grid(True)

    ax2.set_title("Linear acceleration (m/s²)")
    l_ax, = ax2.plot([], [], label="ax")
    l_ay, = ax2.plot([], [], label="ay")
    l_az, = ax2.plot([], [], label="az")
    ax2.legend(loc="upper right"); ax2.grid(True)

    ax3.set_title("Quaternion components")
    l_qw, = ax3.plot([], [], label="qw")
    l_qx, = ax3.plot([], [], label="qx")
    l_qy, = ax3.plot([], [], label="qy")
    l_qz, = ax3.plot([], [], label="qz")
    ax3.legend(loc="upper right"); ax3.grid(True)
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, args.window)

    def on_timer(_frame):
        nonlocal t0
        # drain a few messages per animation tick
        for _ in range(100):
            try:
                pkt = sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            d = parse(pkt)
            if not d:
                continue
            if t0 is None:
                t0 = d["t"]
            tr = d["t"] - t0
            t_rel.append(tr)

            qw, qx, qy, qz = d["quat"]
            gx, gy, gz = d["gyro"]
            ax, ay, az = d["accel"]

            q_w.append(qw); q_x.append(qx); q_y.append(qy); q_z.append(qz)
            g_x.append(gx); g_y.append(gy); g_z.append(gz)
            a_x.append(ax); a_y.append(ay); a_z.append(az)

        if not t_rel:
            return []

        # limit x window
        tmax = t_rel[-1]
        tmin = max(0.0, tmax - args.window)
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(tmin, tmax)

        # update lines
        l_gx.set_data(t_rel, g_x); l_gy.set_data(t_rel, g_y); l_gz.set_data(t_rel, g_z)
        l_ax.set_data(t_rel, a_x); l_ay.set_data(t_rel, a_y); l_az.set_data(t_rel, a_z)
        l_qw.set_data(t_rel, q_w); l_qx.set_data(t_rel, q_x); l_qy.set_data(t_rel, q_y); l_qz.set_data(t_rel, q_z)

        # autoscale y smoothly
        for ax, series in (
            (ax1, (g_x, g_y, g_z)),
            (ax2, (a_x, a_y, a_z)),
            (ax3, (q_w, q_x, q_y, q_z)),
        ):
            ymin = min(min(s) for s in series)
            ymax = max(max(s) for s in series)
            if ymin == ymax:
                ymin -= 1; ymax += 1
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        return [l_gx, l_gy, l_gz, l_ax, l_ay, l_az, l_qw, l_qx, l_qy, l_qz]

    ani = FuncAnimation(fig, on_timer, interval=50, blit=False)
    fig.tight_layout()

    try:
        print(f"Listening on {args.endpoint} …")
        plt.show()
    finally:
        if args.save and t_rel:
            import csv
            with open("imu_log.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t","qw","qx","qy","qz","gx","gy","gz","ax","ay","az"])
                for i in range(len(t_rel)):
                    w.writerow([
                        t_rel[i],
                        q_w[i], q_x[i], q_y[i], q_z[i],
                        g_x[i], g_y[i], g_z[i],
                        a_x[i], a_y[i], a_z[i],
                    ])
            print("Saved imu_log.csv")

if __name__ == "__main__":
    main()
