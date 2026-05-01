"""Main runtime for motor vibration simulation, AI detection, and MQTT publishing."""

import signal
import time
from threading import Lock

from anomaly_detector import AnomalyDetector
from mqtt_client import (
    create_client,
    publish_alert,
    publish_data,
    publish_prediction,
    subscribe_control,
)
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
    motor_state: str,
    detection_method: str = "isolation_forest",
) -> dict:
    """Construct a normalized sensor payload following project schema."""
    return {
        "timestamp": int(time.time()),
        "sensor_id": "motor_01",
        "vibration": round(vibration, 4),
        "unit": "g",
        "status": status,
        "motor_state": motor_state,
        "ai_confidence": confidence,
        "detection_method": detection_method,
    }


def main() -> None:
    """Run the vibration monitoring pipeline with AI anomaly detection."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Initialize AI detector
    print("Initializing AI anomaly detector...")
    detector = AnomalyDetector()

    # Connect MQTT
    print("Connecting to MQTT broker...")
    client = create_client()
    state_lock = Lock()
    motor_state = {"running": True}

    def handle_motor_command(command: dict) -> None:
        requested = str(command.get("motor_state", "")).strip().upper()
        if requested not in {"ON", "OFF"}:
            print(f"Ignoring unsupported motor command: {command}")
            return

        with state_lock:
            if requested == "ON":
                detector.reset()
                motor_state["running"] = True
                print("Motor control command received: ON")
            else:
                motor_state["running"] = False
                print("Motor control command received: OFF")

    subscribe_control(client, handle_motor_command)

    print("Starting vibration monitoring with AI detection...\n")

    for vibration, ground_truth in simulate_stream(lambda: motor_state["running"]):
        if not RUNNING:
            break

        with state_lock:
            running = motor_state["running"]

        if running:
            # AI-based anomaly detection while motor is running
            status, confidence = detector.predict(vibration)
            if status == "FAULT":
                with state_lock:
                    motor_state["running"] = False
                running = False
        else:
            status, confidence = ("MOTOR_OFF", 0.0)

        # Build and publish payload
        motor_label = "ON" if running else "OFF"
        payload = build_payload(vibration, status, confidence, motor_label)
        publish_data(client, payload)
        publish_prediction(client, status, vibration, confidence, motor_label)

        # Publish alert on FAULT
        if status == "FAULT":
            publish_alert(client, status, vibration, confidence, motor_label)

        # Log with ground truth comparison
        gt_label = "FAULT" if ground_truth else "NORMAL"
        match_marker = "+" if (status == gt_label) else "x"
        if status == "MOTOR_OFF":
            gt_label = "MOTOR_OFF"
            match_marker = "+"
        print(
            f"[{match_marker}] Vibration: {vibration:.4f}g | "
            f"AI: {status} (conf: {confidence:.2f}) | "
            f"Motor: {motor_label} | Ground Truth: {gt_label}"
        )

    # Cleanup
    client.loop_stop()
    client.disconnect()
    print("Application stopped.")


if __name__ == "__main__":
    main()
