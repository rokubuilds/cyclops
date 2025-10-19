#!/usr/bin/env python3
"""
SIMPLE GAIT RECOGNITION - Fresh start every time
8 features: speed (avg, std, max, min, range) + height (avg, std, range)
Enroll users, then straight to monitoring
"""

import struct
import zmq
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from collections import deque
import time

# ============================================================================
# DATA STREAM
# ============================================================================

class L2DataStream:
    def __init__(self, ip='192.168.1.249', port=5555):
        self.context = zmq.Context()
        self.pc_socket = self.context.socket(zmq.SUB)
        self.pc_socket.connect(f"tcp://{ip}:{port}")
        self.pc_socket.setsockopt(zmq.SUBSCRIBE, b"")
        print(f"✅ Connected to {ip}:{port}")

    def get_point_cloud(self):
        data = self.pc_socket.recv()
        seq, stamp_sec, stamp_nsec = struct.unpack('<III', data[0:12])
        ring_num, cloud_size = struct.unpack('<II', data[12:20])

        point_data = data[20:]
        points = []
        offset = 0

        for i in range(cloud_size):
            if offset + 22 > len(point_data):
                break
            x, y, z, intensity, point_time = struct.unpack('<fffff', point_data[offset:offset+20])
            ring = struct.unpack('<H', point_data[offset+20:offset+22])[0]
            points.append([x, y, z, intensity])
            offset += 22

        return {
            'timestamp': time.time(),
            'points': np.array(points) if points else np.array([]).reshape(0, 4)
        }

# ============================================================================
# PERSON DETECTION
# ============================================================================

class PersonDetector:
    def __init__(self):
        self.history = deque(maxlen=90)

    def detect_person(self, point_cloud):
        points = point_cloud['points']

        if len(points) < 50:
            return None

        human_points = points[
            (points[:, 2] > -1.5) &
            (points[:, 2] < 1.0) &
            (points[:, 0] > 1.0) &
            (points[:, 0] < 10.0)
        ]

        if len(human_points) < 50:
            return None

        clustering = DBSCAN(eps=0.5, min_samples=30).fit(human_points[:, :3])
        labels = clustering.labels_

        unique = [l for l in set(labels) if l != -1]
        if not unique:
            return None

        cluster_sizes = [(l, np.sum(labels == l)) for l in unique]
        best_label = max(cluster_sizes, key=lambda x: x[1])[0]
        person_points = human_points[labels == best_label]

        height = person_points[:, 2].max() - person_points[:, 2].min()
        if not (0.5 < height < 2.0):
            return None

        center = person_points[:, :3].mean(axis=0)

        return {
            'center': center,
            'height': height,
            'points': person_points
        }

    def update_history(self, person, timestamp):
        if person is None:
            return None

        self.history.append({
            'timestamp': timestamp,
            'center': person['center'],
            'height': person['height']
        })

        return list(self.history)

    def clear(self):
        self.history.clear()

# ============================================================================
# FEATURE EXTRACTION - 8 simple features
# ============================================================================

class GaitFeatures:
    def extract(self, history):
        """Extract 8 simple features - speed and height ONLY"""
        if len(history) < 30:
            return None

        positions = np.array([h['center'] for h in history])
        timestamps = np.array([h['timestamp'] for h in history])
        heights = np.array([h['height'] for h in history])

        # Calculate velocities
        time_diff = np.diff(timestamps)
        if np.any(time_diff <= 0):
            time_diff = np.full(len(timestamps)-1, 0.1)

        velocities = np.diff(positions, axis=0) / time_diff[:, np.newaxis]
        speeds = np.linalg.norm(velocities[:, :2], axis=1)
        speeds = speeds[speeds < 10.0]

        if len(speeds) < 5:
            return None

        # 8 SIMPLE FEATURES ONLY
        features = {
            # Speed (5)
            'avg_speed': float(np.mean(speeds)),
            'speed_std': float(np.std(speeds)),
            'max_speed': float(np.max(speeds)),
            'min_speed': float(np.min(speeds)),
            'speed_range': float(np.ptp(speeds)),

            # Height (3)
            'avg_height': float(np.mean(heights)),
            'height_std': float(np.std(heights)),
            'height_range': float(np.ptp(heights))
        }

        return features

    def feature_vector(self, features):
        """Convert to 8-element vector"""
        if features is None:
            return None
        return np.array([
            # Speed (5)
            features['avg_speed'],
            features['speed_std'],
            features['max_speed'],
            features['min_speed'],
            features['speed_range'],
            # Height (3)
            features['avg_height'],
            features['height_std'],
            features['height_range']
        ])

# ============================================================================
# IDENTIFIER
# ============================================================================

class GaitIdentifier:
    def __init__(self):
        self.users = {}
        self.scaler = StandardScaler()
        # Balanced class weights - prevent bias to one user!
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced')
        self.trained = False
        self.feature_extractor = GaitFeatures()

    def train(self):
        """Train SVM"""
        if len(self.users) < 2:
            print("❌ Need 2+ users to train")
            return False

        print("\n⏳ Training SVM...")

        X, y = [], []
        user_stats = {}  # Track feature statistics per user

        for name, data in self.users.items():
            speeds = []
            heights = []
            for feat in data['samples']:
                vec = self.feature_extractor.feature_vector(feat)
                if vec is not None:
                    X.append(vec)
                    y.append(data['label'])
                    speeds.append(feat['avg_speed'])
                    heights.append(feat['avg_height'])

            user_stats[name] = {
                'speed_mean': np.mean(speeds) if speeds else 0,
                'speed_std': np.std(speeds) if speeds else 0,
                'height_mean': np.mean(heights) if heights else 0,
                'height_std': np.std(heights) if heights else 0
            }

        X = np.array(X)
        y = np.array(y)

        # Print feature statistics
        print("\n📊 Feature Statistics:")
        for name, stats in user_stats.items():
            print(f"   {name}:")
            print(f"      Speed:  {stats['speed_mean']:.3f} ± {stats['speed_std']:.3f} m/s")
            print(f"      Height: {stats['height_mean']:.3f} ± {stats['height_std']:.3f} m")

        if len(X) < 30:
            print(f"❌ Only {len(X)} valid samples (need 30+)")
            return False

        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        print(f"   Total: {len(X)} samples")

        # Train
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True

        acc = np.mean(self.model.predict(X_scaled) == y)
        print(f"\n✅ Trained! Overall accuracy: {acc*100:.1f}%")

        # Per-user accuracy
        predictions = self.model.predict(X_scaled)
        for name, data in self.users.items():
            label = data['label']
            user_mask = y == label
            user_acc = np.mean(predictions[user_mask] == label)
            print(f"   {name}: {user_acc*100:.1f}%")

        return True

    def identify(self, features):
        """Identify from features"""
        if not self.trained:
            return "Not trained", 0.0

        vec = self.feature_extractor.feature_vector(features)
        if vec is None:
            return "Unknown", 0.0

        vec_scaled = self.scaler.transform([vec])
        probs = self.model.predict_proba(vec_scaled)[0]
        pred_label = np.argmax(probs)
        confidence = probs[pred_label]

        # DEBUG: Print all probabilities
        print(f"\nDEBUG - All probs: ", end='')
        for name, data in self.users.items():
            label = data['label']
            print(f"{name}={probs[label]*100:.1f}% ", end='')
        print(f"| Predicted: label={pred_label}")

        # Threshold for 2 users
        threshold = 0.60  # Need majority confidence

        if confidence < threshold:
            return "Unknown", confidence

        for name, data in self.users.items():
            if data['label'] == pred_label:
                return name, confidence

        return "Unknown", 0.0

# ============================================================================
# MAIN SYSTEM
# ============================================================================

class SimpleGaitSystem:
    def __init__(self, ip='192.168.1.249'):
        self.stream = L2DataStream(ip)
        self.detector = PersonDetector()
        self.features = GaitFeatures()
        self.identifier = GaitIdentifier()

    def enroll_user(self, name, target_samples=100):
        """Enroll a user"""
        print(f"\n{'='*60}")
        print(f"👤 ENROLLING: {name}")
        print(f"{'='*60}")
        print(f"   Target: {target_samples} samples")
        print(f"   Walk naturally back and forth")
        input("\nPress Enter to start...")

        samples = []
        rejected = 0
        self.detector.clear()

        print("\n🔴 RECORDING...\n")

        try:
            while len(samples) < target_samples:
                pc = self.stream.get_point_cloud()
                person = self.detector.detect_person(pc)
                history = self.detector.update_history(person, pc['timestamp'])

                if history and len(history) >= 30:
                    feat = self.features.extract(history)

                    if feat is not None:
                        samples.append(feat)
                        progress = len(samples) / target_samples
                        bar = '█' * int(progress * 40) + '░' * (40 - int(progress * 40))
                        total = len(samples) + rejected
                        accept_rate = len(samples) / total * 100 if total > 0 else 0
                        print(f"\r[{bar}] {len(samples)}/{target_samples} | Accept: {accept_rate:.0f}%", end='')
                    else:
                        rejected += 1

        except KeyboardInterrupt:
            print("\n\n❌ Cancelled")
            return False

        print(f"\n\n✅ Collected {len(samples)} samples")

        # Add to identifier
        label = len(self.identifier.users)
        self.identifier.users[name] = {'label': label, 'samples': samples}

        return True

    def monitor(self):
        """Monitor and identify"""
        if not self.identifier.trained:
            print("❌ Model not trained yet!")
            return

        print("\n" + "="*60)
        print("🎬 MONITORING MODE")
        print("="*60)
        print(f"{len(self.identifier.users)} users loaded")
        print("Press Ctrl+C to stop\n")

        self.detector.clear()
        last_pred = time.time()
        frame_count = 0

        try:
            while True:
                pc = self.stream.get_point_cloud()
                detection = self.detector.detect_person(pc)
                history = self.detector.update_history(detection, pc['timestamp'])

                # ONLY predict if person is ACTUALLY detected
                if detection and history and len(self.detector.history) >= 60:
                    if time.time() - last_pred > 1.5:
                        features = self.features.extract(history)

                        # Double check features are valid
                        if features is not None:
                            name, conf = self.identifier.identify(features)

                            if name not in ["Unknown", "Not trained"]:
                                print("\n" + "╔" + "═" * 58 + "╗")
                                print("║" + " " * 58 + "║")
                                print("║" + f"  🎯  IDENTIFIED: {name.upper()}".center(58) + "║")
                                print("║" + f"  Confidence: {conf*100:.0f}%".center(58) + "║")
                                print("║" + " " * 58 + "║")
                                print("╚" + "═" * 58 + "╝")
                            else:
                                print(f"\r❓ Unknown (conf: {conf*100:.0f}%)", end='')

                            last_pred = time.time()

                frame_count += 1
                if frame_count % 50 == 0:
                    status = "✓ Person" if detection else "⚠️  No person"
                    hist_len = len(self.detector.history)
                    print(f"\r{status} | History: {hist_len} frames", end='')

        except KeyboardInterrupt:
            print("\n\n👋 Stopped monitoring")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🔐 SIMPLE GAIT RECOGNITION")
    print("="*60)
    print("14 features: speed, height, accel, vertical, rhythm, sway")
    print("100 samples per user, BALANCED class weights")
    print("Fresh start every run")
    print("="*60)

    system = SimpleGaitSystem(ip='192.168.1.249')

    # Enroll users until we have at least 2
    while len(system.identifier.users) < 2:
        print(f"\n⚠️  Need at least 2 users (currently have {len(system.identifier.users)})")
        name = input("👤 Enter name to enroll (or 'quit' to exit): ").strip()

        if name.lower() == 'quit':
            print("👋 Bye!")
            exit(0)

        if name:
            system.enroll_user(name, target_samples=100)  # Back to 100
        else:
            print("❌ Invalid name")

    # Ask if user wants to enroll more
    while True:
        print(f"\n✅ Currently have {len(system.identifier.users)} user(s):")
        for name in system.identifier.users.keys():
            print(f"   • {name}")

        choice = input("\nEnroll another user? (y/n): ").strip().lower()

        if choice == 'y':
            name = input("👤 Enter name: ").strip()
            if name:
                system.enroll_user(name, target_samples=100)  # Back to 100
            else:
                print("❌ Invalid name")
        else:
            break

    # Train model
    if system.identifier.train():
        # Start monitoring
        system.monitor()
    else:
        print("\n❌ Training failed!")
