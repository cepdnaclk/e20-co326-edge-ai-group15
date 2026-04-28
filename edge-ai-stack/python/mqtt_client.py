"""MQTT client utilities for motor vibration publishing."""

import os
import paho.mqtt.client as mqtt

BROKER_HOST = os.environ.get("BROKER_HOST", "localhost")
BROKER_PORT = int(os.environ.get("BROKER_PORT", "1883"))
GROUP_ID = os.environ.get("GROUP_ID", "group01")

DATA_TOPIC = f"sensors/{GROUP_ID}/motor-vibration/data"
ALERT_TOPIC = f"alerts/{GROUP_ID}/motor-vibration/status"
