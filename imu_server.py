#!/usr/bin/env python3
"""
Flask server to bridge IMU data from ZMQ to web interface
"""
import json
import struct
import time
from collections import deque
from threading import Thread, Lock
import zmq
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# IMU Configuration
IMU_ENDPOINT = "tcp://192.168.1.249:5556"
Q_IS_WXYZ = False
PACK_FMT = "<IIIffffffffff"  # 3*uint32 + 10*float32 = 12 + 40 = 52 bytes
PACK_LEN = 52

# Global data storage
imu_data = {
    'gyro': {'x': deque(maxlen=100), 'y': deque(maxlen=100), 'z': deque(maxlen=100)},
    'accel': {'x': deque(maxlen=100), 'y': deque(maxlen=100), 'z': deque(maxlen=100)},
    'quat': {'w': deque(maxlen=100), 'x': deque(maxlen=100), 'y': deque(maxlen=100), 'z': deque(maxlen=100)},
    'timestamps': deque(maxlen=100),
    'connected': False,
    'last_update': 0
}
data_lock = Lock()

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

def zmq_worker():
    """Worker thread to receive ZMQ data"""
    global imu_data
    
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(IMU_ENDPOINT)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.RCVTIMEO = 1000  # 1 second timeout
    
    print(f"ZMQ worker started, listening on {IMU_ENDPOINT}")
    
    while True:
        try:
            pkt = sub.recv()
            d = parse(pkt)
            if d:
                with data_lock:
                    imu_data['timestamps'].append(d['t'])
                    imu_data['gyro']['x'].append(d['gyro'][0])
                    imu_data['gyro']['y'].append(d['gyro'][1])
                    imu_data['gyro']['z'].append(d['gyro'][2])
                    imu_data['accel']['x'].append(d['accel'][0])
                    imu_data['accel']['y'].append(d['accel'][1])
                    imu_data['accel']['z'].append(d['accel'][2])
                    imu_data['quat']['w'].append(d['quat'][0])
                    imu_data['quat']['x'].append(d['quat'][1])
                    imu_data['quat']['y'].append(d['quat'][2])
                    imu_data['quat']['z'].append(d['quat'][3])
                    imu_data['last_update'] = time.time()
                    imu_data['connected'] = True
        except zmq.Again:
            # Timeout - check if we've lost connection
            with data_lock:
                if time.time() - imu_data['last_update'] > 5.0:
                    imu_data['connected'] = False
        except Exception as e:
            print(f"ZMQ error: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return send_from_directory('.', 'demo.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

@app.route('/api/imu')
def get_imu_data():
    """Get current IMU data"""
    with data_lock:
        # Convert deques to lists for JSON serialization
        data = {
            'connected': imu_data['connected'],
            'gyro': {
                'x': list(imu_data['gyro']['x']),
                'y': list(imu_data['gyro']['y']),
                'z': list(imu_data['gyro']['z'])
            },
            'accel': {
                'x': list(imu_data['accel']['x']),
                'y': list(imu_data['accel']['y']),
                'z': list(imu_data['accel']['z'])
            },
            'quat': {
                'w': list(imu_data['quat']['w']),
                'x': list(imu_data['quat']['x']),
                'y': list(imu_data['quat']['y']),
                'z': list(imu_data['quat']['z'])
            },
            'timestamps': list(imu_data['timestamps'])
        }
    return jsonify(data)

@app.route('/api/status')
def get_status():
    """Get connection status"""
    with data_lock:
        return jsonify({
            'connected': imu_data['connected'],
            'last_update': imu_data['last_update'],
            'data_points': len(imu_data['timestamps'])
        })

if __name__ == '__main__':
    # Start ZMQ worker thread
    zmq_thread = Thread(target=zmq_worker, daemon=True)
    zmq_thread.start()
    
    # Start Flask server
    print("Starting Flask server on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)
