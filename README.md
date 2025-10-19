# CYCLOPS - Gait Analysis Demo

This project integrates IMU sensor data visualization into a web-based gait analysis demo.

## Features

- Real-time IMU data visualization (gyroscope, accelerometer, quaternion)
- Web-based interface with live charts
- Fallback simulation mode when IMU server is not available
- Modern, responsive UI design

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the IMU Server

Start the Flask server that bridges ZMQ IMU data to the web interface:

```bash
python imu_server.py
```

The server will:
- Connect to the IMU data source via ZMQ (default: `tcp://192.168.1.249:5556`)
- Serve the web interface at `http://localhost:8080`
- Provide API endpoints for real-time IMU data

### 3. View the Demo

Open your browser and navigate to:
- **With IMU server running**: `http://localhost:8080` (shows real IMU data)
- **Standalone**: Open `demo.html` directly (shows simulated data)

## File Structure

- `demo.html` - Main web interface with integrated sensor status
- `imu_server.py` - Flask server for bridging IMU data
- `lidar/imu_plot.py` - Original matplotlib-based IMU plotting script
- `requirements.txt` - Python dependencies
- `styles.css` - Styling for the web interface
- `script.js` - JavaScript functionality

## API Endpoints

- `GET /` - Main demo page
- `GET /api/imu` - Current IMU data (JSON)
- `GET /api/status` - Connection status

## Configuration

Edit `imu_server.py` to modify:
- IMU endpoint: `IMU_ENDPOINT = "tcp://192.168.1.249:5556"`
- Data buffer size: `maxlen=100` in deque initialization
- Server port: `app.run(port=8080)`

## Usage

1. **Real IMU Data**: Start the server and ensure your IMU device is streaming data via ZMQ
2. **Simulation Mode**: Open `demo.html` directly in a browser for demo purposes
3. **Hybrid**: The web interface automatically detects if the server is available and switches between real and simulated data

The sensor status section shows three charts:
- **Angular Velocity**: Gyroscope data (gx, gy, gz) in rad/s
- **Linear Acceleration**: Accelerometer data (ax, ay, az) in m/s²
- **Quaternion**: Orientation data (qw, qx, qy, qz)

Connection status is indicated by a colored dot:
- 🟢 Green: Connected to real IMU data
- 🔴 Red: Disconnected or simulation mode
