"""MQTT client utilities for motor vibration publishing."""

import json
import os
import time

import paho.mqtt.client as mqtt

BROKER_HOST = os.environ.get("BROKER_HOST", "localhost")
BROKER_PORT = int(os.environ.get("BROKER_PORT", "1883"))
GROUP_ID = os.environ.get("GROUP_ID", "group01")
DATA_TOPIC = f"sensors/{GROUP_ID}/motor-vibration/data"
ALERT_TOPIC = f"alerts/{GROUP_ID}/motor-vibration/status"


def _on_connect(client: mqtt.Client, userdata, flags, rc):
    """Handle connection callback and log status."""
    if rc == 0:
        print(f"Connected to MQTT broker at {BROKER_HOST}:{BROKER_PORT}")
    else:
        print(f"MQTT connection failed with return code {rc}")


def create_client() -> mqtt.Client:
    """Create and connect an MQTT client with retry logic.

    Returns:
        A connected paho-mqtt client.

    Raises:
        ConnectionError: If connection cannot be established after retries.
    """
    client = mqtt.Client()
    client.on_connect = _on_connect

    retries = 3
    retry_delay_seconds = 5

    for attempt in range(1, retries + 1):
        try:
            print(
                f"Attempting MQTT connection to {BROKER_HOST}:{BROKER_PORT} "
                f"(attempt {attempt}/{retries})"
            )
            client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
            client.loop_start()
            return client
        except Exception as exc:
            print(f"Connection attempt {attempt} failed: {exc}")
            if attempt < retries:
                print(f"Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)

    raise ConnectionError(
        f"Unable to connect to MQTT broker at {BROKER_HOST}:{BROKER_PORT}"
    )


def publish_data(client: mqtt.Client, payload: dict) -> None:
    """Publish sensor data payload to the data topic as JSON."""
    message = json.dumps(payload)
    result = client.publish(DATA_TOPIC, message)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"Failed to publish data message (rc={result.rc})")


def publish_alert(client: mqtt.Client, status: str, vibration_value: float) -> None:
    """Publish FAULT/NORMAL alert payload to the alert topic as JSON."""
    payload = {
        "timestamp": int(time.time()),
        "sensor_id": "motor_01",
        "vibration": round(vibration_value, 4),
        "unit": "g",
        "status": status,
    }
    message = json.dumps(payload)
    result = client.publish(ALERT_TOPIC, message)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"Failed to publish alert message (rc={result.rc})")
