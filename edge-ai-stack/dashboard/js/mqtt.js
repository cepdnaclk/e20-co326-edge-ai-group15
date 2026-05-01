/**
 * mqtt.js — MQTT over WebSocket client
 *
 * Connects to the Mosquitto broker, subscribes to all three topics,
 * and dispatches parsed payloads to the dashboard handler functions.
 *
 * Depends on: mqtt.min.js (CDN), config.js, dashboard.js
 */

let mqttClient = null;

/* ── Connection state ─────────────────────────────────────────────── */
const connBadge = document.getElementById('connBadge');
const connLabel = document.getElementById('connLabel');

function setConnState(state) {
  connBadge.className = 'conn-badge ' + state;
  const labels = {
    connecting: 'Connecting…',
    connected:  'Connected',
    error:      'Disconnected',
    '':         'Offline',
  };
  connLabel.textContent = labels[state] ?? 'Unknown';
}

/* ══════════════════════════════════════════════════════════════════
   CONNECT / RECONNECT
   ══════════════════════════════════════════════════════════════════ */
function mqttConnect(brokerUrl, groupId) {
  // Tear down any existing connection cleanly
  if (mqttClient) {
    try { mqttClient.end(true); } catch (_) {}
    mqttClient = null;
  }

  // Update config from UI inputs if provided
  if (brokerUrl) CONFIG.BROKER_URL = brokerUrl;
  if (groupId)   CONFIG.GROUP_ID   = groupId;

  setConnState('connecting');
  logEvent('info', `Connecting to ${CONFIG.BROKER_URL} …`);

  const clientId = `dashboard_${Math.random().toString(16).slice(2, 8)}`;

  mqttClient = mqtt.connect(CONFIG.BROKER_URL, {
    clientId,
    keepalive:       30,
    reconnectPeriod: 5000,
    connectTimeout:  10_000,
    clean:           true,
  });

  mqttClient.on('connect', () => {
    setConnState('connected');
    logEvent('info', `Connected · client ${clientId}`);

    const topics = CONFIG.TOPICS;
    const toSubscribe = [topics.data, topics.alert, topics.prediction];

    toSubscribe.forEach(topic => {
      mqttClient.subscribe(topic, { qos: 0 }, (err) => {
        if (err) {
          logEvent('error', `Subscribe failed: ${topic}`);
        } else {
          logEvent('info', `Subscribed: ${topic}`);
        }
      });
    });
  });

  mqttClient.on('reconnect', () => {
    setConnState('connecting');
    logEvent('info', 'Reconnecting…');
  });

  mqttClient.on('error', (err) => {
    setConnState('error');
    logEvent('error', `MQTT error: ${err.message}`);
  });

  mqttClient.on('offline', () => {
    setConnState('error');
    logEvent('error', 'Broker unreachable');
  });

  mqttClient.on('close', () => {
    setConnState('error');
  });

  /* ── Message dispatcher ─────────────────────────────────────── */
  mqttClient.on('message', (topic, payloadBuf) => {
    let payload;
    try {
      payload = JSON.parse(payloadBuf.toString());
    } catch {
      logEvent('error', `Malformed JSON on topic: ${topic}`);
      return;
    }

    const topics = CONFIG.TOPICS;
    if (topic === topics.prediction || topic === topics.data) {
      handlePrediction(payload, topic);
    }
    if (topic === topics.alert) {
      handleAlert(payload);
    }
  });
}

/* ── Kick off initial connection ────────────────────────────────── */
mqttConnect();
