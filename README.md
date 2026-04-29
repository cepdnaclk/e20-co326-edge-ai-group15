# Motor Vibration Monitoring — Edge AI IoT Project

## Group Members
- E/20/248 - Mapagedara T.L.B.
- E/20/453 - Yogesh R.J.
- E/20/158 - Jananga T.G.C.
- E/20/300 - Prasadinie H.A.M.T.

## Project Description
This project implements a complete end-to-end Edge AI and Industrial IoT workflow for motor vibration monitoring. A Python application simulates motor vibration signals, applies real-time anomaly detection, and publishes structured telemetry and alerts to MQTT topics. Node-RED subscribes to those topics and presents a live dashboard with trend charts, intensity gauge visualization, and clear fault status messaging.

## System Architecture
Sensor Simulator → Anomaly Detector → MQTT Publisher → Broker → Node-RED → Dashboard

## How to Run
1. Clone the repository.
2. (Optional) Edit broker configuration in `python/mqtt_client.py` or set environment variables:
   - `BROKER_HOST` (default: `localhost`)
   - `BROKER_PORT` (default: `1883`)
   - `GROUP_ID` (default: `group01`)
3. Start all services:
   ```bash
   docker-compose up --build
   ```
4. Open Node-RED at `http://localhost:1880`.
5. Dashboard should appear at `http://localhost:1880/ui` after flows load.
6. To stop services:
   ```bash
   docker-compose down
   ```

## MQTT Topics Used
| Topic | Purpose |
|-------|---------|
| sensors/group01/motor-vibration/data | Real-time sensor readings |
| alerts/group01/motor-vibration/status | Fault alerts only |

## Results
- Add screenshots of:
  - Live chart behavior during normal operation
  - Gauge changes under fault spikes
  - Status panel showing `⚠ FAULT DETECTED`


