# Motor Vibration Monitoring — Edge AI IoT Project

![Edge AI](https://img.shields.io/badge/AI-Edge_Computing-blue)
![IoT](https://img.shields.io/badge/IoT-MQTT_Node--RED-orange)
![Machine Learning](https://img.shields.io/badge/ML-Isolation_Forest-brightgreen)

## 👥 Group Members
- **E/20/248** - Mapagedara T.L.B.
- **E/20/453** - Yogesh R.J.
- **E/20/158** - Jananga T.G.C.
- **E/20/300** - Prasadinie H.A.M.T.

# Motor Vibration Anomaly Detection System

An end-to-end IoT pipeline for real-time motor health monitoring using AI-powered anomaly detection. The system simulates vibration sensor data, detects faults using a trained Isolation Forest model, publishes results via MQTT, and visualizes them through a Node-RED dashboard.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Quick Start (Docker)](#quick-start-docker)
- [Local Development Setup](#local-development-setup)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [MQTT Topics](#mqtt-topics)
- [Node-RED Dashboard](#node-red-dashboard)
- [Testing](#testing)
- [Configuration](#configuration)
- [Feature Engineering](#feature-engineering)
- [Model Details](#model-details)

---

## Architecture Overview

```
┌─────────────────────┐     MQTT      ┌──────────────┐     ┌─────────────────┐
│  Python Application │ ──────────── ▶│  Mosquitto   │◀─── │   Node-RED      │
│                     │               │  MQTT Broker │     │   Dashboard     │
│  ┌───────────────┐  │               └──────────────┘     └─────────────────┘
│  │  Vibration    │  │
│  │  Simulator    │  │   Topics:
│  └──────┬────────┘  │   • sensors/group01/motor-vibration/data
│         │           │   • alerts/group01/motor-vibration/status
│  ┌──────▼────────┐  │   • ai/group01/motor-vibration/prediction
│  │  Anomaly      │  │
│  │  Detector     │  │
│  │ (Isolation    │  │
│  │  Forest)      │  │
│  └───────────────┘  │
└─────────────────────┘
```

The system is composed of three Docker services:

| Service | Description |
|---|---|
| `mosquitto` | Eclipse Mosquitto MQTT broker (ports 1883 / 9001 WebSocket) |
| `python-app` | Vibration simulator + AI anomaly detector + MQTT publisher |
| `nodered` | Node-RED dashboard for real-time visualization |

---

## Project Structure

```
project/
│
├── python/                        # Python application
│   ├── main.py                    # Main runtime entry point
│   ├── vibration_simulator.py     # Synthetic vibration signal generator
│   ├── anomaly_detector.py        # Runtime AI inference wrapper
│   ├── generate_dataset.py        # Training dataset generator
│   ├── train_model.py             # Model training pipeline
│   ├── mqtt_client.py             # MQTT publish utilities
│   ├── requirements.txt           # Full dependencies (dev + test)
│   ├── requirements_docker.txt    # Minimal runtime dependencies
│   ├── Dockerfile                 # Python app container
│   │
│   ├── dataset/
│   │   └── vibration_data.csv     # Pre-generated training dataset (9,991 rows)
│   │
│   ├── model/                     # Trained model artifacts (generated)
│   │   ├── anomaly_detector.joblib
│   │   └── scaler.joblib
│   │
│   └── tests/
│       ├── test_anomaly_detector.py
│       ├── test_generate_dataset.py
│       ├── test_train_model.py
│       └── test_integration.py
│
├── node-red/                      # Node-RED configuration
│   ├── flows.json                 # Dashboard flow definitions
│   ├── settings.js                # Node-RED settings
│   ├── package.json               # Node-RED dependencies
│   └── Dockerfile                 # Node-RED container
│
├── mosquitto/
│   └── mosquitto.conf             # MQTT broker configuration
│
└── docker-compose.yml             # Full stack orchestration
```

---

## How It Works

### 1. Vibration Simulation
`vibration_simulator.py` generates synthetic motor vibration readings (in g-units) at 1-second intervals:

- **Normal mode**: Low-amplitude sinusoidal oscillation with sensor noise (typically 0.0 – 0.35 g)
- **Fault mode** (~15% probability): Either a sharp impulse spike (0.8 – 2.0 g) or an unstable resonance burst

### 2. Feature Engineering
Each raw reading is enriched with a sliding-window feature vector before inference:

| Feature | Description |
|---|---|
| `vibration` | Raw sensor reading (g) |
| `rolling_mean` | Mean over the last N readings |
| `rolling_std` | Standard deviation over the last N readings |
| `rolling_min` | Minimum value in the window |
| `rolling_max` | Maximum value in the window |
| `delta` | Absolute difference from the previous reading |

The window size defaults to **10 readings**.

### 3. AI Anomaly Detection
An **Isolation Forest** model (scikit-learn) classifies each feature vector:

- `NORMAL` → inlier (Isolation Forest label: `1`)
- `FAULT` → outlier (Isolation Forest label: `-1`)

A sigmoid mapping converts the raw anomaly score into a 0–1 confidence value.

### 4. MQTT Publishing
Results are published to three topics on each reading cycle (see [MQTT Topics](#mqtt-topics)).

### 5. Motor Safety Logic
When a `FAULT` is detected, the motor is flagged as `OFF` and the simulator stops generating active readings until the application is restarted.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/)
- Python 3.11+ (for local development only)

---

## Quick Start (Docker)

```bash
# Clone the repository
git clone <repo-url>
cd <repo-directory>

# Build and start all services
docker compose up --build

# Access the Node-RED dashboard
open http://localhost:1880/ui
```

All three services start automatically. The Python app connects to Mosquitto and begins streaming data within seconds.

To stop the stack:
```bash
docker compose down
```

---

## Local Development Setup

### 1. Install Python dependencies

```bash
cd python
pip install -r requirements.txt
```

### 2. Start the MQTT broker

You can run Mosquitto locally or use the Docker service:

```bash
docker compose up mosquitto
```

### 3. Generate a training dataset

```bash
python generate_dataset.py
# Outputs: dataset/vibration_data.csv  (~10,000 rows)
```

### 4. Train the model

```bash
python train_model.py
# Outputs: model/anomaly_detector.joblib
#          model/scaler.joblib
```

### 5. Run the application

```bash
BROKER_HOST=localhost python main.py
```

---

## Training the Model

The training pipeline can be customized via CLI arguments:

```bash
python train_model.py \
  --dataset dataset/vibration_data.csv \
  --output  model/anomaly_detector.joblib \
  --scaler  model/scaler.joblib
```

**Training configuration** (edit `train_model.py` to change defaults):

| Parameter | Default | Description |
|---|---|---|
| `CONTAMINATION` | `0.15` | Expected anomaly fraction |
| `N_ESTIMATORS` | `150` | Number of Isolation Forest trees |
| `TEST_SIZE` | `0.20` | Fraction of data held out for evaluation |
| `RANDOM_STATE` | `42` | Reproducibility seed |

After training, the script prints a full classification report and confusion matrix.

### Dataset Generation

```bash
python generate_dataset.py
```

Configurable parameters in `generate_dataset.py`:

| Parameter | Default | Description |
|---|---|---|
| `TOTAL_SAMPLES` | `10,000` | Total rows to generate |
| `FAULT_RATIO` | `0.15` | Proportion of fault samples |
| `WINDOW_SIZE` | `10` | Sliding window length |

---

## Running the Application

```bash
python main.py
```

The application will:
1. Load the trained model from `model/`
2. Connect to the MQTT broker (with retry logic, 3 attempts)
3. Begin streaming vibration readings at 1-second intervals
4. Print live results to stdout with ground-truth comparison markers

Example output:
```
[+] Vibration: 0.1423g | AI: NORMAL (conf: 0.23) | Motor: ON | Ground Truth: NORMAL
[x] Vibration: 1.5341g | AI: NORMAL (conf: 0.48) | Motor: ON | Ground Truth: FAULT
[+] Vibration: 1.2104g | AI: FAULT  (conf: 0.81) | Motor: ON | Ground Truth: FAULT
```

Graceful shutdown is supported via `Ctrl+C` (SIGINT) or SIGTERM.

---

## MQTT Topics

All payloads are JSON. The `GROUP_ID` environment variable (default: `group01`) is interpolated into topic names.

### `sensors/{GROUP_ID}/motor-vibration/data`
Published every cycle. Full sensor reading with AI result.

```json
{
  "timestamp": 1718000000,
  "sensor_id": "motor_01",
  "vibration": 0.1423,
  "unit": "g",
  "status": "NORMAL",
  "motor_state": "ON",
  "ai_confidence": 0.2341,
  "detection_method": "isolation_forest"
}
```

### `alerts/{GROUP_ID}/motor-vibration/status`
Published **only on FAULT events**.

```json
{
  "timestamp": 1718000000,
  "sensor_id": "motor_01",
  "vibration": 1.5341,
  "unit": "g",
  "status": "FAULT",
  "motor_state": "ON",
  "ai_confidence": 0.8102,
  "detection_method": "isolation_forest"
}
```

### `ai/{GROUP_ID}/motor-vibration/prediction`
Published every cycle. AI-focused payload without units metadata.

```json
{
  "timestamp": 1718000000,
  "sensor_id": "motor_01",
  "vibration": 0.1423,
  "status": "NORMAL",
  "motor_state": "ON",
  "ai_confidence": 0.2341,
  "detection_method": "isolation_forest"
}
```

---

## Node-RED Dashboard

Access at **http://localhost:1880/ui** after starting the stack.

The dashboard has four panels:

| Panel | Widgets | Description |
|---|---|---|
| **Vibration Metrics** | Line chart + Gauge | Real-time vibration (g) over time |
| **Fault Status** | Text indicator | Live status with AI confidence |
| **AI Prediction** | Confidence gauge + Text | Isolation Forest output |
| **Motor Status** | Text indicator | Motor ON/OFF state |

The Node-RED editor is available at **http://localhost:1880**.

---

## Testing

Run the full test suite from the `python/` directory:

```bash
pytest -v
```

### Test coverage

| File | Scope |
|---|---|
| `test_generate_dataset.py` | Unit tests for rolling statistics helpers and CSV generation |
| `test_train_model.py` | End-to-end training pipeline: file creation, metrics, model persistence, and prediction sanity checks |
| `test_anomaly_detector.py` | Detector initialization, warm-up period, NORMAL/FAULT classification accuracy |
| `test_integration.py` | Full pipeline: generate → train → detect; payload builder; vibration simulator ranges |

Run a specific test file:
```bash
pytest test_anomaly_detector.py -v
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BROKER_HOST` | `localhost` | MQTT broker hostname |
| `BROKER_PORT` | `1883` | MQTT broker port |
| `GROUP_ID` | `group01` | Topic namespace identifier |
| `PYTHONUNBUFFERED` | — | Set to `1` in Docker for real-time logs |

### MQTT Broker

The Mosquitto broker is configured in `mosquitto/mosquitto.conf`:
- Port `1883` — standard MQTT
- Port `9001` — WebSocket (for browser clients)
- Anonymous connections allowed
- No message persistence

---

## Feature Engineering

The detector replicates the exact feature computation used during training to avoid train/serve skew. Features are computed over a configurable sliding window of recent readings:

```
window = [v₁, v₂, ..., vₙ]  (n = window_size, default 10)

rolling_mean = mean(window)
rolling_std  = population_std(window)
rolling_min  = min(window)
rolling_max  = max(window)
delta        = |current - previous|
```

During the **warm-up period** (first `window_size - 1` readings), the detector returns `("NORMAL", 0.0)` as there is insufficient data to compute stable rolling statistics.

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Isolation Forest |
| Library | scikit-learn |
| Contamination | 15% (matches simulator fault rate) |
| Trees | 150 |
| Feature scaling | StandardScaler (zero mean, unit variance) |
| Serialization | joblib |
| Max model size | < 5 MB (suitable for edge deployment) |

**Confidence scoring**: The raw Isolation Forest `decision_function` score (more negative = more anomalous) is mapped to [0, 1] via a sigmoid:

```
confidence = 1 / (1 + exp(5 × raw_score))
```

Higher confidence values indicate higher certainty of a fault.