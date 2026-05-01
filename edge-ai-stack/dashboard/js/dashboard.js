/**
 * dashboard.js — Core UI logic
 *
 * Handles:
 *  • MQTT payload processing
 *  • KPI card updates
 *  • Rolling stats display
 *  • Event log
 *  • Fault overlay
 *  • Config panel
 *  • Clock & session uptime
 */

/* ══════════════════════════════════════════════════════════════════
   STATE
   ══════════════════════════════════════════════════════════════════ */
const state = {
  faultCount:   0,
  messageCount: 0,
  sessionStart: Date.now(),
  lastStatus:   null,
  showThreshold: true,
};

/* ══════════════════════════════════════════════════════════════════
   DOM REFS
   ══════════════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);

const dom = {
  kpiVib:       $('kpiVibVal'),
  kpiStatus:    $('kpiStatusVal'),
  kpiStatusSub: $('kpiStatusSub'),
  kpiConf:      $('kpiConfVal'),
  kpiMotor:     $('kpiMotorVal'),
  kpiMotorSub:  $('kpiMotorSub'),
  kpiFault:     $('kpiFaultVal'),
  kpiUptime:    $('kpiUptimeVal'),

  kpiVibCard:    $('kpiVibration'),
  kpiStatusCard: $('kpiStatus'),
  kpiConfCard:   $('kpiConfidence'),
  kpiMotorCard:  $('kpiMotor'),
  kpiFaultCard:  $('kpiFaults'),

  statMean:     $('statMean'),
  statStd:      $('statStd'),
  statMin:      $('statMin'),
  statMax:      $('statMax'),
  statDelta:    $('statDelta'),
  statMsgCount: $('statMsgCount'),

  sensorId:       $('sensorId'),
  sensorUnit:     $('sensorUnit'),
  sensorLastSeen: $('sensorLastSeen'),
  sensorTopic:    $('sensorTopic'),
  methodBadge:    $('methodBadge'),

  log:          $('eventLog'),
  clock:        $('clock'),

  faultOverlay: $('faultOverlay'),
  faultDetail:  $('faultDetail'),
  dismissFault: $('dismissFault'),

  clearChart:       $('clearChart'),
  clearLog:         $('clearLog'),
  toggleFaultLine:  $('toggleFaultLine'),

  configPanel:  $('configPanel'),
  openConfig:   $('openConfig'),
  cfgBroker:    $('cfgBroker'),
  cfgGroup:     $('cfgGroup'),
  cfgConnect:   $('cfgConnect'),
};

/* ══════════════════════════════════════════════════════════════════
   CLOCK & UPTIME
   ══════════════════════════════════════════════════════════════════ */
function padZ(n) { return String(n).padStart(2, '0'); }

function formatUptime(ms) {
  const s  = Math.floor(ms / 1000);
  const h  = Math.floor(s / 3600);
  const m  = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return h ? `${padZ(h)}:${padZ(m)}:${padZ(ss)}` : `${padZ(m)}:${padZ(ss)}`;
}

setInterval(() => {
  const now = new Date();
  dom.clock.textContent = `${padZ(now.getHours())}:${padZ(now.getMinutes())}:${padZ(now.getSeconds())}`;
  dom.kpiUptime.textContent = formatUptime(Date.now() - state.sessionStart);
}, 1000);

/* ══════════════════════════════════════════════════════════════════
   KPI HELPERS
   ══════════════════════════════════════════════════════════════════ */
function setKpiClass(card, cls) {
  card.classList.remove('ok', 'fault', 'warn');
  if (cls) card.classList.add(cls);
}

function flash(el) {
  el.classList.remove('updated');
  void el.offsetWidth; // reflow to restart animation
  el.classList.add('updated');
}

/* ══════════════════════════════════════════════════════════════════
   TIMESTAMP LABEL (for chart x-axis)
   ══════════════════════════════════════════════════════════════════ */
function tsLabel(unixSeconds) {
  const d = new Date((unixSeconds || Date.now() / 1000) * 1000);
  return `${padZ(d.getHours())}:${padZ(d.getMinutes())}:${padZ(d.getSeconds())}`;
}

/* ══════════════════════════════════════════════════════════════════
   HANDLE PREDICTION / DATA PAYLOAD
   Topic: ai/{group}/motor-vibration/prediction
          sensors/{group}/motor-vibration/data
   ══════════════════════════════════════════════════════════════════ */
function handlePrediction(payload, topic) {
  state.messageCount++;

  const vibration   = parseFloat(payload.vibration   ?? 0);
  const confidence  = parseFloat(payload.ai_confidence ?? 0);
  const status      = String(payload.status ?? 'NORMAL');
  const motorState  = String(payload.motor_state ?? 'ON');
  const method      = String(payload.detection_method ?? 'isolation_forest');
  const sensorId    = String(payload.sensor_id ?? '—');
  const unit        = String(payload.unit ?? 'g');
  const ts          = payload.timestamp;
  const label       = tsLabel(ts);

  const isFault = status === 'FAULT';
  const isMotorOff = motorState === 'OFF' || status === 'MOTOR_OFF';

  /* ── KPI: Vibration ─────────────────────────────────────────── */
  dom.kpiVib.textContent = vibration.toFixed(4);
  flash(dom.kpiVib);
  setKpiClass(dom.kpiVibCard, isFault ? 'fault' : vibration >= 0.6 ? 'warn' : 'ok');

  /* ── KPI: Status ────────────────────────────────────────────── */
  if (status === 'MOTOR_OFF' || isMotorOff) {
    dom.kpiStatus.textContent = 'MOTOR OFF';
    dom.kpiStatusSub.textContent = 'Motor stopped after fault';
    setKpiClass(dom.kpiStatusCard, 'fault');
  } else if (isFault) {
    dom.kpiStatus.textContent = 'FAULT';
    dom.kpiStatusSub.textContent = `Conf: ${(confidence * 100).toFixed(1)}%`;
    setKpiClass(dom.kpiStatusCard, 'fault');
  } else {
    dom.kpiStatus.textContent = 'NORMAL';
    dom.kpiStatusSub.textContent = `Conf: ${(confidence * 100).toFixed(1)}%`;
    setKpiClass(dom.kpiStatusCard, 'ok');
  }
  flash(dom.kpiStatus);

  /* ── KPI: Confidence ────────────────────────────────────────── */
  dom.kpiConf.textContent = (confidence * 100).toFixed(1);
  flash(dom.kpiConf);
  setKpiClass(dom.kpiConfCard, isFault ? 'fault' : confidence > 0.5 ? 'warn' : 'ok');

  /* ── KPI: Motor ─────────────────────────────────────────────── */
  dom.kpiMotor.textContent = motorState;
  dom.kpiMotorSub.textContent = isMotorOff ? 'Relay tripped' : 'Running';
  setKpiClass(dom.kpiMotorCard, isMotorOff ? 'fault' : 'ok');
  flash(dom.kpiMotor);

  /* ── KPI: Faults ────────────────────────────────────────────── */
  if (isFault) {
    state.faultCount++;
    dom.kpiFault.textContent = state.faultCount;
    flash(dom.kpiFault);
    setKpiClass(dom.kpiFaultCard, 'fault');
  }

  /* ── Charts ─────────────────────────────────────────────────── */
  pushVibrationPoint(label, vibration, isFault, state.showThreshold);
  updateGauge(vibration, status);
  pushConfidencePoint(label, confidence, isFault);

  /* ── Rolling stats (only from data topic which has full payload) */
  if (payload.rolling_mean !== undefined) {
    dom.statMean.textContent  = parseFloat(payload.rolling_mean).toFixed(4);
    dom.statStd.textContent   = parseFloat(payload.rolling_std).toFixed(4);
    dom.statMin.textContent   = parseFloat(payload.rolling_min).toFixed(4);
    dom.statMax.textContent   = parseFloat(payload.rolling_max).toFixed(4);
    dom.statDelta.textContent = parseFloat(payload.delta).toFixed(4);
  }
  dom.statMsgCount.textContent = state.messageCount;

  /* ── Sensor meta ─────────────────────────────────────────────── */
  dom.sensorId.textContent       = sensorId;
  dom.sensorUnit.textContent     = unit;
  dom.sensorLastSeen.textContent = label;
  dom.sensorTopic.textContent    = topic.split('/').slice(-2).join('/');
  dom.methodBadge.textContent    = method;

  /* ── Event log ──────────────────────────────────────────────── */
  if (state.lastStatus !== status || isFault) {
    appendLog(label, status, `${vibration.toFixed(4)}g · conf ${(confidence * 100).toFixed(1)}%`);
    state.lastStatus = status;
  }
}

/* ══════════════════════════════════════════════════════════════════
   HANDLE ALERT PAYLOAD
   Topic: alerts/{group}/motor-vibration/status
   ══════════════════════════════════════════════════════════════════ */
function handleAlert(payload) {
  const vibration  = parseFloat(payload.vibration ?? 0);
  const confidence = parseFloat(payload.ai_confidence ?? 0);
  const ts         = tsLabel(payload.timestamp);

  // Show fault overlay
  dom.faultDetail.innerHTML =
    `Sensor: ${payload.sensor_id ?? '—'}<br>` +
    `Vibration: ${vibration.toFixed(4)} g<br>` +
    `Confidence: ${(confidence * 100).toFixed(1)}%<br>` +
    `Time: ${ts}`;

  dom.faultOverlay.classList.add('visible');

  // Auto-dismiss if configured
  if (CONFIG.FAULT_OVERLAY_AUTO_DISMISS_S > 0) {
    setTimeout(() => dom.faultOverlay.classList.remove('visible'),
      CONFIG.FAULT_OVERLAY_AUTO_DISMISS_S * 1000);
  }
}

/* ══════════════════════════════════════════════════════════════════
   EVENT LOG
   ══════════════════════════════════════════════════════════════════ */
function appendLog(time, status, detail) {
  // Remove placeholder
  const placeholder = dom.log.querySelector('.log-placeholder');
  if (placeholder) placeholder.remove();

  const isFault = status === 'FAULT' || status === 'MOTOR_OFF';
  const cls     = isFault ? 'fault' : 'normal';

  const entry = document.createElement('div');
  entry.className = `log-entry ${cls}`;
  entry.innerHTML =
    `<span class="log-entry__time">${time}</span>` +
    `<span class="log-entry__status">${status}</span>` +
    `<span class="log-entry__detail">${detail}</span>`;

  // Prepend so newest is at top
  dom.log.insertBefore(entry, dom.log.firstChild);

  // Trim old entries
  while (dom.log.children.length > CONFIG.LOG_MAX_ENTRIES) {
    dom.log.removeChild(dom.log.lastChild);
  }
}

/**
 * logEvent — log system/connection events (not vibration data)
 * Called by mqtt.js for connection state changes.
 */
function logEvent(level, message) {
  const now = new Date();
  const label = `${padZ(now.getHours())}:${padZ(now.getMinutes())}:${padZ(now.getSeconds())}`;
  appendLog(label, level.toUpperCase(), message);
}

/* ══════════════════════════════════════════════════════════════════
   EVENT LISTENERS
   ══════════════════════════════════════════════════════════════════ */

// Dismiss fault overlay
dom.dismissFault.addEventListener('click', () => {
  dom.faultOverlay.classList.remove('visible');
});

// Clear chart button
dom.clearChart.addEventListener('click', () => {
  clearCharts();
});

// Clear log button
dom.clearLog.addEventListener('click', () => {
  dom.log.innerHTML = '<div class="log-placeholder">Log cleared</div>';
});

// Threshold toggle
dom.toggleFaultLine.addEventListener('change', e => {
  state.showThreshold = e.target.checked;
  setThresholdVisible(state.showThreshold);
});

// Config panel open/close
dom.openConfig.addEventListener('click', () => {
  dom.configPanel.classList.toggle('open');
});
document.addEventListener('click', e => {
  if (!dom.configPanel.contains(e.target) && e.target !== dom.openConfig) {
    dom.configPanel.classList.remove('open');
  }
});

// Config connect button
dom.cfgConnect.addEventListener('click', () => {
  const url   = dom.cfgBroker.value.trim();
  const group = dom.cfgGroup.value.trim();
  if (!url) return;
  dom.configPanel.classList.remove('open');
  mqttConnect(url, group);
});
dom.cfgBroker.addEventListener('keydown', e => {
  if (e.key === 'Enter') dom.cfgConnect.click();
});
