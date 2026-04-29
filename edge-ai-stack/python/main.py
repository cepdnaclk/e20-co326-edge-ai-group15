"""Main runtime for motor vibration simulation, AI detection, and MQTT publishing."""

import signal
import time

from anomaly_detector import AnomalyDetector
from mqtt_client import create_client, publish_alert, publish_data
from vibration_simulator import simulate_stream

RUNNING = True


def handle_shutdown(signum, frame) -> None:
    """Handle SIGINT/SIGTERM and request graceful shutdown."""
    global RUNNING
    RUNNING = False
    print("\nShutdown signal received. Stopping application...")


def build_payload(
    vibration: float,
    status: str,
    confidence: float,
    detection_method: str = "isolation_forest",
) -> dict:
    """Construct a normalized sensor payload following project schema."""
    return {
        "timestamp": int(time.time()),
        "sensor_id": "motor_01",
        "vibration": round(vibration, 4),
        "unit": "g",
        "status": status,
        "ai_confidence": confidence,
        "detection_method": detection_method,
    }


def main() -> None:
    """Run the vibration simulation and publish results until shutdown."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    detector = AnomalyDetector()
    client = create_client()
    stream = simulate_stream()

    try:
        while RUNNING:
            vibration = next(stream)
            status = detector.detect(vibration)
            payload = build_payload(vibration, status)

            publish_data(client, payload)
            publish_alert(client, status, vibration)
            print(
                f"Published vibration={payload['vibration']}g "
                f"status={status} threshold={detector.threshold}"
            )
    finally:
        client.loop_stop()
        client.disconnect()
        print("MQTT client disconnected.")


if __name__ == "__main__":
    main()