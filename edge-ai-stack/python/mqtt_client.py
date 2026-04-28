"""MQTT client utilities for motor vibration publishing."""

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
    """Create and connect an MQTT client with retry logic."""
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