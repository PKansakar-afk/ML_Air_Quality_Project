// ═══════════════════════════════════════════════════════════
//  AQI Intelligence — main.js
// ═══════════════════════════════════════════════════════════

// Active Chart instances (destroyed before re-render)
let bucketChartNowcast = null;
let bucketChartHist    = null;
let heatmapTrendChart  = null;

const MONTH_NAMES = [
  "January","February","March","April","May","June",
  "July","August","September","October","November","December"
];

// ── Tab switching ────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => {
      p.classList.remove("active");
      p.classList.add("hidden");
    });
    btn.classList.add("active");
    const panel = document.getElementById("tab-" + btn.dataset.tab);
    panel.classList.remove("hidden");
    panel.classList.add("active");
  });
});

// ── Utility: AQI → colour ────────────────────────────────
function aqiColor(aqi) {
  if (aqi <= 50)  return "#00c853";
  if (aqi <= 100) return "#aeea00";
  if (aqi <= 200) return "#ffd600";
  if (aqi <= 300) return "#ff6d00";
  if (aqi <= 400) return "#dd2c00";
  return "#6a0080";
}

function aqiBucket(aqi) {
  if (aqi <= 50)  return "Good";
  if (aqi <= 100) return "Satisfactory";
  if (aqi <= 200) return "Moderate";
  if (aqi <= 300) return "Poor";
  if (aqi <= 400) return "Very Poor";
  return "Severe";
}

// Clamp AQI to 0–500 for percentage calculations
function aqiToPercent(aqi) {
  return Math.min(100, Math.max(0, aqi / 5));
}

// ── Bucket donut chart ───────────────────────────────────
const BUCKET_COLORS = {
  "Good":         "#00c853",
  "Satisfactory": "#aeea00",
  "Moderate":     "#ffd600",
  "Poor":         "#ff6d00",
  "Very Poor":    "#dd2c00",
  "Severe":       "#6a0080",
};

function renderBucketChart(canvasId, bucketDist, existingChart) {
  if (existingChart) existingChart.destroy();
  const canvas = document.getElementById(canvasId);
  if (!canvas || !bucketDist || !bucketDist.length) return null;

  const labels = bucketDist.map(b => b.AQI_Bucket);
  const data   = bucketDist.map(b => b.pct);
  const colors = labels.map(l => BUCKET_COLORS[l] || "#888");

  return new Chart(canvas, {
    type: "doughnut",
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 1, borderColor: "#07101f" }] },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "right",
          labels: { color: "#8aaccc", font: { family: "Jost", size: 11 }, boxWidth: 12, padding: 10 }
        },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%` }
        }
      },
      cutout: "65%",
    }
  });
}

// ═══════════════════════════════════════════════════════════
//  TAB 1 — NOWCAST
// ═══════════════════════════════════════════════════════════
async function runNowcast() {
  const city = document.getElementById("nowcast-city").value;
  if (!city) return;

  // Show loading
  document.getElementById("nowcast-results").classList.add("hidden");
  document.getElementById("nowcast-loading").classList.remove("hidden");

  try {
    const res  = await fetch("/api/nowcast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ city })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderNowcast(data);
  } catch (err) {
    document.getElementById("nowcast-loading").classList.add("hidden");
    alert("Error: " + err.message);
  }
}

function renderNowcast(data) {
  document.getElementById("nowcast-loading").classList.add("hidden");
  document.getElementById("nowcast-results").classList.remove("hidden");

  // ── Live data strip ────────────────────────────────────
  const live    = data.live;
  const waqi    = live.waqi || {};
  const weather = live.weather || {};

  document.getElementById("live-station-name").textContent =
    waqi.station ? `${waqi.station}` : `${data.city} (estimated)`;
  document.getElementById("live-updated").textContent =
    waqi.updated ? `Updated: ${waqi.updated}` : "";

  const pollutantDefs = [
    { key: "aqi",     label: "AQI" },
    { key: "pm25",    label: "PM2.5" },
    { key: "pm10",    label: "PM10" },
    { key: "no2",     label: "NO₂" },
    { key: "so2",     label: "SO₂" },
    { key: "co",      label: "CO" },
  ];
  const weatherDefs = [
    { key: "temp",     label: "Temp °C" },
    { key: "humidity", label: "Humidity %" },
    { key: "wind",     label: "Wind km/h" },
  ];

  const liveGrid = document.getElementById("live-pollutants");
  liveGrid.innerHTML = "";

  [...pollutantDefs, ...weatherDefs].forEach(def => {
    const src = pollutantDefs.includes(def) ? waqi : weather;
    const val = src[def.key];
    const el  = document.createElement("div");
    el.className = "live-item";
    el.innerHTML = `
      <span class="live-item-label">${def.label}</span>
      <span class="live-item-value">${val != null ? val : "—"}</span>
    `;
    liveGrid.appendChild(el);
  });

  // ── Model cards ────────────────────────────────────────
  const MODEL_ORDER = ["Random Forest", "XGBoost", "LightGBM", "Ridge"];
  const MODEL_COLORS = {
    "Random Forest": "#FF9800",
    "XGBoost":       "#2196F3",
    "LightGBM":      "#4CAF50",
    "Ridge":         "#AB47BC",
  };

  const grid = document.getElementById("model-cards");
  grid.innerHTML = "";
  MODEL_ORDER.forEach((name, i) => {
    const pred = data.predictions[name] || {};
    const aqi  = pred.aqi;
    const col  = aqi != null ? aqiColor(aqi) : "#555";
    const card = document.createElement("div");
    card.className = "model-card";
    card.style.animationDelay = `${i * 0.08}s`;
    card.style.setProperty("--model-color", MODEL_COLORS[name]);
    card.innerHTML = `
      <div class="mc-name">${name}</div>
      <div class="mc-aqi" style="color:${col}">${aqi != null ? aqi : "—"}</div>
      <div class="mc-bucket" style="color:${col}; border:1px solid ${col}30">
        ${pred.bucket || "N/A"}
      </div>
    `;
    grid.appendChild(card);
  });

  // ── Historical context ────────────────────────────────
  document.getElementById("ctx-month-name").textContent = data.month_name;
  const hist = data.historical;
  if (hist && hist.mean) {
    const wrap = document.getElementById("range-bar-wrap");
    const pMin = aqiToPercent(hist.min);
    const pMax = aqiToPercent(hist.max);
    const pMean= aqiToPercent(hist.mean);
    wrap.innerHTML = `
      <div class="range-track">
        <div class="range-marker" style="left:${pMean}%"></div>
      </div>
    `;
    document.getElementById("range-stats").innerHTML = `
      <div class="rs-item"><span class="rs-label">Min</span><span class="rs-value" style="color:var(--good)">${hist.min}</span></div>
      <div class="rs-item"><span class="rs-label">Avg</span><span class="rs-value" style="color:var(--accent)">${hist.mean}</span></div>
      <div class="rs-item"><span class="rs-label">Max</span><span class="rs-value" style="color:var(--verypoor)">${hist.max}</span></div>
    `;
  }

  // ── Bucket donut chart ────────────────────────────────
  bucketChartNowcast = renderBucketChart("bucket-chart-nowcast", data.bucket_dist, bucketChartNowcast);

  // ── API warnings ──────────────────────────────────────
  const warnBox = document.getElementById("api-warnings");
  const warns   = [];
  if (live.waqi_err)    warns.push(`⚠ WAQI: ${live.waqi_err}. Predictions use historical medians for pollutant values.`);
  if (live.weather_err) warns.push(`⚠ Weather: ${live.weather_err}. Predictions use historical medians for weather.`);
  if (warns.length) {
    warnBox.innerHTML = warns.join("<br>");
    warnBox.classList.remove("hidden");
  } else {
    warnBox.classList.add("hidden");
  }
}

// ═══════════════════════════════════════════════════════════
//  TAB 2 — HISTORICAL LOOKUP
// ═══════════════════════════════════════════════════════════
async function lookupHistorical() {
  const city  = document.getElementById("hist-city").value;
  const month = document.getElementById("hist-month").value;
  const day   = document.getElementById("hist-day").value;

  document.getElementById("hist-results").classList.add("hidden");
  document.getElementById("hist-loading").classList.remove("hidden");

  try {
    const url = `/api/historical?city=${encodeURIComponent(city)}&month=${month}&day=${day}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderHistorical(data, city, parseInt(month), parseInt(day));
  } catch (err) {
    document.getElementById("hist-loading").classList.add("hidden");
    alert("Error: " + err.message);
  }
}

function renderHistorical(data, city, month, day) {
  document.getElementById("hist-loading").classList.add("hidden");
  document.getElementById("hist-results").classList.remove("hidden");

  const label = data.level === "day"
    ? `${city} — ${MONTH_NAMES[month-1]} ${day}`
    : `${city} — ${MONTH_NAMES[month-1]} (month average)`;
  document.getElementById("hist-label").textContent = label;

  const color = data.color || "#00d4ff";
  document.getElementById("hist-mean").textContent = data.mean || "—";
  document.getElementById("hist-min").textContent  = data.min  || "—";
  document.getElementById("hist-max").textContent  = data.max  || "—";

  // Bucket badge
  const badge = document.getElementById("hist-bucket-badge");
  badge.textContent = data.bucket || "";
  badge.style.background = (data.color || "#333") + "30";
  badge.style.color       = data.color || "#ccc";
  badge.style.border      = `1px solid ${data.color || "#333"}50`;

  // Insight text
  const bd        = data.bucket_dist || [];
  const topBucket = bd.length ? bd.reduce((a,b) => a.pct > b.pct ? a : b) : null;
  const insight   = document.getElementById("hist-insight");
  if (topBucket) {
    insight.textContent =
      `Historically, ${topBucket.pct}% of days in ${MONTH_NAMES[month-1]} for ${city} ` +
      `are classified as "${topBucket.AQI_Bucket}". ` +
      `The typical AQI range spans from ${data.min} (cleanest) to ${data.max} (worst recorded).`;
  }

  // Range visualisation
  const MAX_AQI  = 500;
  const minPct   = (data.min  / MAX_AQI * 100).toFixed(1);
  const maxPct   = (data.max  / MAX_AQI * 100).toFixed(1);
  const meanPct  = (data.mean / MAX_AQI * 100).toFixed(1);

  const vizWrap = document.getElementById("hist-range-viz");
  vizWrap.innerHTML = `
    <div class="range-viz" style="margin:0">
      <div class="viz-pointer" style="left:${minPct}%; border-top-color:var(--good)"></div>
      <div class="viz-pointer" style="left:${meanPct}%; border-top-color:#fff"></div>
      <div class="viz-pointer" style="left:${maxPct}%; border-top-color:var(--verypoor)"></div>
    </div>
    <div class="viz-labels">
      <span>0</span><span>100</span><span>200</span><span>300</span><span>400</span><span>500+</span>
    </div>
    <div style="display:flex; gap:1.5rem; margin-top:0.6rem; font-size:0.8rem; color:var(--text-dim)">
      <span style="color:var(--good)">▲ Min: ${data.min}</span>
      <span style="color:#fff">▲ Avg: ${data.mean}</span>
      <span style="color:var(--verypoor)">▲ Max: ${data.max}</span>
    </div>
  `;

  // Donut chart
  bucketChartHist = renderBucketChart("bucket-chart-hist", data.bucket_dist, bucketChartHist);
}

// ═══════════════════════════════════════════════════════════
//  TAB 3 — RISK HEATMAP
// ═══════════════════════════════════════════════════════════
async function loadHeatmap() {
  const city = document.getElementById("heatmap-city").value;

  document.getElementById("heatmap-results").classList.add("hidden");
  document.getElementById("heatmap-loading").classList.remove("hidden");

  try {
    const res  = await fetch(`/api/heatmap?city=${encodeURIComponent(city)}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderHeatmap(data);
  } catch (err) {
    document.getElementById("heatmap-loading").classList.add("hidden");
    alert("Error: " + err.message);
  }
}

function renderHeatmap(data) {
  document.getElementById("heatmap-loading").classList.add("hidden");
  document.getElementById("heatmap-results").classList.remove("hidden");

  document.getElementById("heatmap-city-title").textContent =
    `${data.city} — Seasonal AQI Risk`;

  const strip   = document.getElementById("month-strip");
  const tooltip = document.getElementById("month-tooltip");
  strip.innerHTML = "";

  data.months.forEach((m, idx) => {
    const cell = document.createElement("div");
    cell.className = "month-cell";
    cell.style.animationDelay = `${idx * 0.05}s`;

    if (m.mean != null) {
      const col = m.color || aqiColor(m.mean);
      cell.style.background  = col + "30";
      cell.style.borderColor = col + "60";
      cell.innerHTML = `
        <div class="mc-month-name" style="color:${col}">${m.name}</div>
        <div class="mc-aqi-value" style="color:${col}">${m.mean}</div>
        <div class="mc-risk-label" style="color:${col}">${m.bucket}</div>
      `;

      // Tooltip on hover
      cell.addEventListener("mouseenter", e => {
        const bd  = (m.bucket_dist || [])
          .sort((a,b) => b.pct - a.pct)
          .slice(0, 3)
          .map(b => `<div class="tooltip-row"><span>${b.AQI_Bucket}</span><span>${b.pct}%</span></div>`)
          .join("");
        tooltip.innerHTML = `
          <div class="tooltip-title">${m.full_name}</div>
          <div class="tooltip-row"><span>Avg AQI</span><span>${m.mean}</span></div>
          <div class="tooltip-row"><span>Min / Max</span><span>${m.min} / ${m.max}</span></div>
          <div class="tooltip-row"><span>Dominant</span><span>${m.dominant}</span></div>
          <hr style="border-color:rgba(255,255,255,0.1); margin:0.4rem 0">
          ${bd}
          ${m.hazard_pct > 0
            ? `<div class="tooltip-row" style="color:#ff6d00"><span>Hazard days</span><span>${m.hazard_pct}%</span></div>`
            : ""}
        `;
        tooltip.classList.remove("hidden");
        positionTooltip(e);
      });
      cell.addEventListener("mousemove", positionTooltip);
      cell.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));
    } else {
      cell.innerHTML = `<div class="mc-month-name" style="color:var(--text-muted)">${m.name}</div>
                        <div style="color:var(--text-muted); font-size:0.8rem">N/A</div>`;
      cell.style.opacity = "0.4";
    }

    strip.appendChild(cell);
  });

  // ── Trend line chart ──────────────────────────────────
  if (heatmapTrendChart) heatmapTrendChart.destroy();
  const trendCanvas = document.getElementById("heatmap-trend-chart");
  const validMonths = data.months.filter(m => m.mean != null);

  heatmapTrendChart = new Chart(trendCanvas, {
    type: "line",
    data: {
      labels: validMonths.map(m => m.name),
      datasets: [{
        label: "Avg AQI",
        data:  validMonths.map(m => m.mean),
        borderColor: ctx => {
          const grad = ctx.chart.ctx.createLinearGradient(0, 0, ctx.chart.width, 0);
          grad.addColorStop(0,   "#00c853");
          grad.addColorStop(0.4, "#ffd600");
          grad.addColorStop(0.8, "#dd2c00");
          return grad;
        },
        backgroundColor: "rgba(0,212,255,0.06)",
        borderWidth: 2.5,
        pointBackgroundColor: validMonths.map(m => m.color || aqiColor(m.mean)),
        pointRadius: 5, pointHoverRadius: 7,
        fill: true, tension: 0.4,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` AQI: ${ctx.parsed.y} (${aqiBucket(ctx.parsed.y)})`
          }
        }
      },
      scales: {
        x: { ticks: { color: "#5a7a99", font: { family: "Jost" } }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: {
          ticks: { color: "#5a7a99", font: { family: "JetBrains Mono", size: 11 } },
          grid:  { color: "rgba(255,255,255,0.04)" },
          min: 0,
        }
      }
    }
  });

  // ── Insight text ──────────────────────────────────────
  const withData = data.months.filter(m => m.mean != null);
  if (withData.length) {
    const worst  = withData.reduce((a,b) => a.mean > b.mean ? a : b);
    const best   = withData.reduce((a,b) => a.mean < b.mean ? a : b);
    const highRisk = withData.filter(m => m.mean > 200).map(m => m.full_name);
    const insightEl = document.getElementById("heatmap-insight");
    insightEl.innerHTML = `
      <p>
        <strong>Worst month:</strong> ${worst.full_name} (avg AQI ${worst.mean} — ${worst.bucket})<br>
        <strong>Best month:</strong>  ${best.full_name}  (avg AQI ${best.mean}  — ${best.bucket})<br>
        ${highRisk.length
          ? `<strong>High-risk months (AQI > 200):</strong> ${highRisk.join(", ")} — consider avoiding outdoor activity.<br>`
          : `No months average above AQI 200 — relatively good annual air quality.<br>`}
        ${worst.hazard_pct > 0
          ? `<strong>${worst.full_name}</strong> has ${worst.hazard_pct}% of days classified as Very Poor or Severe.`
          : ""}
      </p>
    `;
  }
}

function positionTooltip(e) {
  const tt = document.getElementById("month-tooltip");
  const margin = 12;
  let x = e.clientX + margin;
  let y = e.clientY + margin;
  if (x + 240 > window.innerWidth)  x = e.clientX - 240 - margin;
  if (y + 200 > window.innerHeight) y = e.clientY - 200 - margin;
  tt.style.left = x + "px";
  tt.style.top  = y + "px";
}