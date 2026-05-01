/**
 * config.js — Application configuration & constants
 * Edit BROKER_URL and GROUP_ID to match your deployment.
 */

const CONFIG = {
  // ── MQTT ────────────────────────────────────────────────────────
  // WebSocket URL of the Mosquitto broker (port 9001 by default).
  // When running via Docker, point this to your Docker host IP.
  BROKER_URL: 'ws://localhost:9001',

  GROUP_ID: 'group01',

  // Derived MQTT topic subscriptions
  get TOPICS() {
    return {
      data:       `sensors/${this.GROUP_ID}/motor-vibration/data`,
      alert:      `alerts/${this.GROUP_ID}/motor-vibration/status`,
      prediction: `ai/${this.GROUP_ID}/motor-vibration/prediction`,
    };
  },

  // ── CHART ────────────────────────────────────────────────────────
  // Maximum data points visible on the vibration & confidence charts
  CHART_MAX_POINTS: 60,

  // Fault threshold line on the vibration chart (g)
  FAULT_THRESHOLD_G: 0.8,

  // ── LOG ──────────────────────────────────────────────────────────
  // Maximum log entries shown before oldest are trimmed
  LOG_MAX_ENTRIES: 200,

  // ── UI ───────────────────────────────────────────────────────────
  // How many seconds after a fault before the overlay auto-dismisses (0 = never)
  FAULT_OVERLAY_AUTO_DISMISS_S: 0,
};
