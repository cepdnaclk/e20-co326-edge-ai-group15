/**
 * charts.js — Chart.js initialization for:
 *   • Vibration time-series line chart
 *   • Doughnut gauge (current intensity)
 *   • AI Confidence history bar chart
 */

/* ── Shared chart defaults ────────────────────────────────────────── */
Chart.defaults.color          = '#666';
Chart.defaults.borderColor    = '#2a2a2a';
Chart.defaults.font.family    = "'IBM Plex Mono', monospace";
Chart.defaults.font.size      = 10;
Chart.defaults.animation      = { duration: 120 };

/* ══════════════════════════════════════════════════════════════════
   VIBRATION LINE CHART
   ══════════════════════════════════════════════════════════════════ */
const vibrationCtx = document.getElementById('vibrationChart').getContext('2d');

const vibrationChart = new Chart(vibrationCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'Vibration (g)',
        data: [],
        borderColor: '#c8c8c8',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 3,
        pointHoverBackgroundColor: '#c8c8c8',
        tension: 0.3,
        fill: {
          target: 'origin',
          above: 'rgba(200,200,200,0.04)',
          below: 'rgba(200,200,200,0.04)',
        },
      },
      {
        // Fault readings — highlighted red dots
        label: 'Fault',
        data: [],
        borderColor: 'transparent',
        backgroundColor: '#b05050',
        pointRadius: 4,
        pointStyle: 'circle',
        showLine: false,
      },
      {
        // Threshold reference line
        label: 'Threshold',
        data: [],
        borderColor: '#5a2828',
        borderWidth: 1,
        borderDash: [5, 4],
        pointRadius: 0,
        tension: 0,
        fill: false,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1a1a1a',
        borderColor: '#333',
        borderWidth: 1,
        titleColor: '#888',
        bodyColor: '#e8e8e8',
        callbacks: {
          label: ctx => {
            if (ctx.datasetIndex === 2) return null; // hide threshold tooltip
            return `${ctx.dataset.label}: ${(ctx.parsed.y ?? 0).toFixed(4)} g`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: { color: '#1f1f1f' },
        ticks: {
          maxTicksLimit: 8,
          maxRotation: 0,
          color: '#555',
        },
      },
      y: {
        min: 0,
        max: 2.2,
        grid: { color: '#1f1f1f' },
        ticks: {
          stepSize: 0.4,
          color: '#555',
          callback: v => v.toFixed(1),
        },
      },
    },
  },
});

/* ══════════════════════════════════════════════════════════════════
   GAUGE (DOUGHNUT)
   ══════════════════════════════════════════════════════════════════ */
const gaugeCtx = document.getElementById('gaugeChart').getContext('2d');

const gaugeChart = new Chart(gaugeCtx, {
  type: 'doughnut',
  data: {
    datasets: [{
      data: [0, 2.0],
      backgroundColor: ['#5a9e6f', '#1a1a1a'],
      borderWidth: 0,
      circumference: 180,
      rotation: 270,
    }],
  },
  options: {
    responsive: false,
    cutout: '72%',
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    animation: { duration: 200 },
  },
});

/* ── Gauge threshold zone rings ────────────────────────────────────
   We draw a custom overlay on the canvas to show the warn/fault zones.
   This runs once after the chart renders.                              */
function drawGaugeZones() {
  const canvas = document.getElementById('gaugeChart');
  const ctx = canvas.getContext('2d');
  const cx = canvas.width / 2;
  const cy = canvas.height * 0.95;
  const outerR = (canvas.width / 2) * 0.88;
  const innerR = outerR * 0.72;

  // Three zones: 0–0.6 (ok), 0.6–1.0 (warn), 1.0–2.0 (fault)
  const total = 2.0;
  const drawArc = (start, end, color) => {
    const startAngle = Math.PI + (start / total) * Math.PI;
    const endAngle   = Math.PI + (end / total) * Math.PI;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, outerR, startAngle, endAngle);
    ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = color + '28'; // translucent
    ctx.fill();
    ctx.restore();
  };

  drawArc(0,   0.6, '#5a9e6f');
  drawArc(0.6, 1.0, '#9e8a4a');
  drawArc(1.0, 2.0, '#b05050');
}

gaugeChart.options.animation.onComplete = drawGaugeZones;

/* ══════════════════════════════════════════════════════════════════
   CONFIDENCE HISTORY (BAR)
   ══════════════════════════════════════════════════════════════════ */
const confCtx = document.getElementById('confidenceChart').getContext('2d');

const confidenceChart = new Chart(confCtx, {
  type: 'bar',
  data: {
    labels: [],
    datasets: [{
      label: 'AI Confidence',
      data: [],
      backgroundColor: [],   // filled dynamically
      borderWidth: 0,
      borderRadius: 1,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1a1a1a',
        borderColor: '#333',
        borderWidth: 1,
        titleColor: '#888',
        bodyColor: '#e8e8e8',
        callbacks: {
          label: ctx => `Confidence: ${(ctx.parsed.y * 100).toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { maxTicksLimit: 8, maxRotation: 0, color: '#555' },
      },
      y: {
        min: 0,
        max: 1,
        grid: { color: '#1f1f1f' },
        ticks: {
          stepSize: 0.2,
          color: '#555',
          callback: v => `${(v * 100).toFixed(0)}%`,
        },
      },
    },
  },
});

/* ══════════════════════════════════════════════════════════════════
   PUBLIC CHART UPDATE FUNCTIONS
   ══════════════════════════════════════════════════════════════════ */

/**
 * Push a new vibration reading onto the chart.
 * @param {string} label     - x-axis timestamp label
 * @param {number} value     - vibration reading in g
 * @param {boolean} isFault  - whether AI flagged this as FAULT
 * @param {boolean} showThreshold
 */
function pushVibrationPoint(label, value, isFault, showThreshold) {
  const maxPts = CONFIG.CHART_MAX_POINTS;
  const ds = vibrationChart.data;

  // Main line
  ds.labels.push(label);
  ds.datasets[0].data.push(value);

  // Fault overlay points (null keeps the point invisible when normal)
  ds.datasets[1].data.push(isFault ? value : null);

  // Threshold line
  ds.datasets[2].data.push(showThreshold ? CONFIG.FAULT_THRESHOLD_G : null);

  // Trim old data
  if (ds.labels.length > maxPts) {
    ds.labels.shift();
    ds.datasets[0].data.shift();
    ds.datasets[1].data.shift();
    ds.datasets[2].data.shift();
  }

  vibrationChart.update('none');
}

/**
 * Update the doughnut gauge with a new value (0 – 2.0 g).
 * @param {number} value
 * @param {string} status - 'NORMAL' | 'FAULT' | other
 */
function updateGauge(value, status) {
  const clamped = Math.min(Math.max(value, 0), 2.0);
  const remaining = 2.0 - clamped;

  let color = '#5a9e6f';
  if (value >= 1.0)      color = '#b05050';
  else if (value >= 0.6) color = '#9e8a4a';

  gaugeChart.data.datasets[0].data = [clamped, remaining];
  gaugeChart.data.datasets[0].backgroundColor = [color, '#1a1a1a'];
  gaugeChart.update('none');

  document.getElementById('gaugeValue').textContent = value.toFixed(4);
  document.getElementById('gaugeValue').style.color = color;
  drawGaugeZones();
}

/**
 * Push a confidence reading onto the bar chart.
 * @param {string} label
 * @param {number} confidence - 0.0 – 1.0
 * @param {boolean} isFault
 */
function pushConfidencePoint(label, confidence, isFault) {
  const maxPts = CONFIG.CHART_MAX_POINTS;
  const ds = confidenceChart.data;

  ds.labels.push(label);
  ds.datasets[0].data.push(confidence);

  // Color bar by severity
  let color;
  if (isFault)             color = '#b05050';
  else if (confidence > 0.5) color = '#9e8a4a';
  else                       color = '#3d5c45';

  ds.datasets[0].backgroundColor.push(color);

  if (ds.labels.length > maxPts) {
    ds.labels.shift();
    ds.datasets[0].data.shift();
    ds.datasets[0].backgroundColor.shift();
  }

  confidenceChart.update('none');
}

/** Clear both time-series charts */
function clearCharts() {
  [vibrationChart, confidenceChart].forEach(c => {
    c.data.labels = [];
    c.data.datasets.forEach(ds => {
      ds.data = [];
      if (ds.backgroundColor && Array.isArray(ds.backgroundColor)) ds.backgroundColor = [];
    });
    c.update('none');
  });
  updateGauge(0, 'NORMAL');
}

/** Toggle the threshold reference line visibility */
function setThresholdVisible(visible) {
  vibrationChart.data.datasets[2].borderColor = visible ? '#5a2828' : 'transparent';
  vibrationChart.update('none');
}
