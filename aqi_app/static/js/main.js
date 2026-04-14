// ═══════════════════════════════════════════════════════
//  AQI Intelligence — main.js
// ═══════════════════════════════════════════════════════

let hcBucketChart  = null;
let tsMainChart    = null;
let tsScatterChart = null;
let tsResidualChart= null;

const MONTH_NAMES = [
  "January","February","March","April","May","June",
  "July","August","September","October","November","December"
];

const MODEL_ORDER  = ["Random Forest","XGBoost","LightGBM","Ridge"];
const MODEL_COLORS = {
  "Random Forest": "#FF9800",
  "XGBoost":       "#2196F3",
  "LightGBM":      "#4CAF50",
  "Ridge":         "#AB47BC",
};

// ── Tab switching ────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => {
      p.classList.remove("active"); p.classList.add("hidden");
    });
    btn.classList.add("active");
    const panel = document.getElementById("tab-" + btn.dataset.tab);
    panel.classList.remove("hidden");
    panel.classList.add("active");
  });
});

// ── Safe fetch (always parses JSON, never throws on HTML) ─
async function safeFetch(url, opts) {
  const res  = await fetch(url, opts);
  const text = await res.text();
  try {
    return { ok: res.ok, data: JSON.parse(text) };
  } catch {
    const preview = text.slice(0, 120).replace(/<[^>]+>/g,"").trim();
    return { ok: false, data: { error: `Server error (${res.status}): ${preview}` } };
  }
}

// ── AQI helpers ──────────────────────────────────────────
function aqiColor(v) {
  if (v <= 50)  return "#00c853";
  if (v <= 100) return "#aeea00";
  if (v <= 200) return "#ffd600";
  if (v <= 300) return "#ff6d00";
  if (v <= 400) return "#dd2c00";
  return "#6a0080";
}
function aqiBucket(v) {
  if (v <= 50)  return "Good";
  if (v <= 100) return "Satisfactory";
  if (v <= 200) return "Moderate";
  if (v <= 300) return "Poor";
  if (v <= 400) return "Very Poor";
  return "Severe";
}
function aqiToPercent(v) { return Math.min(100, Math.max(0, v / 5)); }

// ── Donut chart helper ───────────────────────────────────
const BUCKET_COLORS = {
  "Good":"#00c853","Satisfactory":"#aeea00","Moderate":"#ffd600",
  "Poor":"#ff6d00","Very Poor":"#dd2c00","Severe":"#6a0080",
};
function renderDonut(canvasId, bucketDist, existing) {
  if (existing) existing.destroy();
  const canvas = document.getElementById(canvasId);
  if (!canvas || !bucketDist?.length) return null;
  return new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: bucketDist.map(b => b.AQI_Bucket),
      datasets: [{
        data:            bucketDist.map(b => b.pct),
        backgroundColor: bucketDist.map(b => BUCKET_COLORS[b.AQI_Bucket] || "#888"),
        borderWidth: 1, borderColor: "#07101f"
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position:"right", labels:{color:"#8aaccc",font:{family:"Jost",size:11},boxWidth:12,padding:10}},
        tooltip: { callbacks:{ label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%` }}
      },
      cutout: "65%",
    }
  });
}

// ══════════════════════════════════════════════════════════
//  TAB 1 — HINDCAST
// ══════════════════════════════════════════════════════════
async function runHindcast() {
  const city = document.getElementById("hc-city").value;
  const date = document.getElementById("hc-date").value;
  if (!city || !date) return;

  document.getElementById("hc-results").classList.add("hidden");
  document.getElementById("hc-loading").classList.remove("hidden");

  const { ok, data } = await safeFetch("/api/hindcast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ city, date })
  });

  document.getElementById("hc-loading").classList.add("hidden");
  if (!ok || data.error) { alert("Error: " + data.error); return; }

  renderHindcast(data);
}

function renderHindcast(data) {
  document.getElementById("hc-results").classList.remove("hidden");

  // ── Actual AQI banner ──────────────────────────────────
  const actual = data.actual;
  const banner = document.getElementById("actual-banner");
  if (actual.aqi != null) {
    banner.classList.remove("hidden");
    banner.style.background  = actual.color + "18";
    banner.style.borderColor = actual.color + "50";
    banner.style.color       = actual.color;

    const errRows = MODEL_ORDER.map(name => {
      const pred = data.predictions[name];
      if (!pred?.aqi) return "";
      const err = ((pred.aqi - actual.aqi) / actual.aqi * 100).toFixed(1);
      return `<span>${name}: ${err > 0 ? "+" : ""}${err}%</span>`;
    }).filter(Boolean).join(" &nbsp;|&nbsp; ");

    banner.innerHTML = `
      <div class="actual-banner-left">
        <span class="actual-banner-label">Actual Recorded AQI — ${data.date}</span>
        <span class="actual-banner-aqi">${actual.aqi}</span>
        <span class="actual-banner-bucket">${actual.bucket}</span>
      </div>
      <div class="actual-banner-right">
        <div style="margin-bottom:0.3rem;opacity:0.6;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px">Model Error vs Actual</div>
        ${errRows}
      </div>`;
  } else {
    banner.classList.add("hidden");
  }

  // ── Data source strip ──────────────────────────────────
  document.getElementById("hc-source-icon").textContent =
    data.air_source === "CPCB Dataset" ? "📂" : "🌐";
  document.getElementById("hc-source-label").textContent =
    `Source: ${data.air_source}`;
  document.getElementById("hc-date-label").textContent = data.date;

  const readings = actual.readings || {};
  const fields   = [
    {k:"pm25",l:"PM2.5 μg/m³"},{k:"pm10",l:"PM10 μg/m³"},
    {k:"no2", l:"NO₂ μg/m³"}, {k:"so2", l:"SO₂ μg/m³"},
    {k:"co",  l:"CO mg/m³"},  {k:"temp",l:"Temp °C"},
    {k:"humidity",l:"Humidity %"},{k:"wind",l:"Wind km/h"},
  ];
  const grid = document.getElementById("hc-readings");
  grid.innerHTML = "";
  fields.forEach(f => {
    const el = document.createElement("div");
    el.className = "live-item";
    el.innerHTML = `
      <span class="live-item-label">${f.l}</span>
      <span class="live-item-value">${readings[f.k] != null ? readings[f.k] : "—"}</span>`;
    grid.appendChild(el);
  });

  // ── Model cards ────────────────────────────────────────
  const mcGrid = document.getElementById("hc-model-cards");
  mcGrid.innerHTML = "";
  MODEL_ORDER.forEach((name, i) => {
    const pred = data.predictions[name] || {};
    const col  = pred.aqi != null ? aqiColor(pred.aqi) : "#555";
    const card = document.createElement("div");
    card.className = "model-card";
    card.style.animationDelay = `${i * 0.08}s`;
    card.innerHTML = `
      <div class="mc-name">${name}</div>
      <div class="mc-aqi" style="color:${col}">${pred.aqi ?? "—"}</div>
      <div class="mc-bucket" style="color:${col};border:1px solid ${col}30">
        ${pred.bucket || "N/A"}
      </div>`;
    mcGrid.appendChild(card);
  });

  // ── Historical range ───────────────────────────────────
  document.getElementById("hc-month-name").textContent = data.month_name;
  const hist = data.historical;
  if (hist?.mean) {
    document.getElementById("hc-range-bar").innerHTML = `
      <div class="range-track">
        <div class="range-marker" style="left:${aqiToPercent(hist.mean)}%"></div>
      </div>`;
    document.getElementById("hc-range-stats").innerHTML = `
      <div class="rs-item"><span class="rs-label">Min</span><span style="color:var(--good);font-family:'JetBrains Mono',monospace">${hist.min}</span></div>
      <div class="rs-item"><span class="rs-label">Avg</span><span style="color:var(--accent);font-family:'JetBrains Mono',monospace">${hist.mean}</span></div>
      <div class="rs-item"><span class="rs-label">Max</span><span style="color:var(--verypoor);font-family:'JetBrains Mono',monospace">${hist.max}</span></div>`;
  }

  hcBucketChart = renderDonut("hc-bucket-chart", data.bucket_dist, hcBucketChart);
}

// ══════════════════════════════════════════════════════════
//  TAB 2 — TIME SERIES COMPARISON
// ══════════════════════════════════════════════════════════
async function loadTimeSeries() {
  const city = document.getElementById("ts-city").value;
  const year = document.getElementById("ts-year").value;
  if (!city || !year) return;

  document.getElementById("ts-results").classList.add("hidden");
  document.getElementById("ts-loading").classList.remove("hidden");

  const { ok, data } = await safeFetch(
    `/api/timeseries?city=${encodeURIComponent(city)}&year=${year}`
  );

  document.getElementById("ts-loading").classList.add("hidden");
  if (!ok || data.error) { alert("Error: " + data.error); return; }

  renderTimeSeries(data);
}

function renderTimeSeries(data) {
  document.getElementById("ts-results").classList.remove("hidden");

  const s = data.stats;

  // ── Stats strip ────────────────────────────────────────
  document.getElementById("ts-stats-strip").innerHTML = `
    <div class="stat-card">
      <div class="stat-label">RMSE</div>
      <div class="stat-value" style="color:#FF9800">${s.rmse}</div>
      <div class="stat-sub">Root Mean Sq. Error</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">MAE</div>
      <div class="stat-value" style="color:#FF9800">${s.mae}</div>
      <div class="stat-sub">Mean Abs. Error</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">R²</div>
      <div class="stat-value" style="color:${s.r2 > 0.9 ? "var(--good)" : "var(--moderate)"}">${s.r2 ?? "—"}</div>
      <div class="stat-sub">Explained Variance</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Avg AQI</div>
      <div class="stat-value" style="color:${aqiColor(s.act_mean)}">${s.act_mean}</div>
      <div class="stat-sub">Min ${s.act_min} · Max ${s.act_max}</div>
    </div>`;

  document.getElementById("ts-chart-title").textContent =
    `${data.city} ${data.year} — Actual vs RF Predicted AQI`;

  // ── Main line chart ────────────────────────────────────
  if (tsMainChart) tsMainChart.destroy();
  tsMainChart = new Chart(document.getElementById("ts-main-chart"), {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "Actual AQI",
          data:  data.actual,
          borderColor: "#00d4ff",
          backgroundColor: "rgba(0,212,255,0.04)",
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
          tension: 0.3,
          spanGaps: true,
        },
        {
          label: "RF Predicted",
          data:  data.predicted,
          borderColor: "#FF9800",
          backgroundColor: "rgba(255,152,0,0.04)",
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: false,
          tension: 0.3,
          borderDash: [4, 2],
        }
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: false   // we use custom legend in HTML
        },
        tooltip: {
          callbacks: {
            title: ctx => ctx[0].label,
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y} (${aqiBucket(ctx.parsed.y)})`
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: "#5a7a99",
            font: { family: "JetBrains Mono", size: 10 },
            maxTicksLimit: 12,
            maxRotation: 0,
          },
          grid: { color: "rgba(255,255,255,0.03)" }
        },
        y: {
          ticks: { color: "#5a7a99", font: { family: "JetBrains Mono", size: 11 } },
          grid:  { color: "rgba(255,255,255,0.04)" },
          min: 0,
        }
      }
    }
  });

  // ── Monthly cards ──────────────────────────────────────
  const mgrid = document.getElementById("ts-monthly-grid");
  mgrid.innerHTML = "";
  data.monthly.forEach((m, i) => {
    const card = document.createElement("div");
    card.className = "month-card";
    card.style.animationDelay = `${i * 0.04}s`;
    if (m.actual_mean != null) {
      const col = m.color || aqiColor(m.actual_mean);
      card.style.background  = col + "22";
      card.style.borderColor = col + "55";
      card.innerHTML = `
        <div class="mc-mname" style="color:${col}">${m.name}</div>
        <div class="mc-maqi"  style="color:${col}">${m.actual_mean}</div>
        <div class="mc-mpred" style="color:${col}">RF:${m.pred_mean}</div>`;
    } else {
      card.style.opacity = "0.3";
      card.innerHTML = `<div class="mc-mname" style="color:var(--text-muted)">${m.name}</div>
                        <div style="color:var(--text-muted);font-size:0.8rem">—</div>`;
    }
    mgrid.appendChild(card);
  });

  // ── Scatter: predicted vs actual ──────────────────────
  if (tsScatterChart) tsScatterChart.destroy();
  const scatterPts = data.actual
    .map((a, i) => ({ x: a, y: data.predicted[i] }))
    .filter(p => p.x != null);

  const maxVal = Math.max(...scatterPts.map(p => Math.max(p.x, p.y)));
  tsScatterChart = new Chart(document.getElementById("ts-scatter-chart"), {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Predicted vs Actual",
          data: scatterPts,
          backgroundColor: "rgba(255,152,0,0.35)",
          pointRadius: 2.5,
          pointHoverRadius: 5,
        },
        {
          label: "Perfect fit",
          data: [{ x:0, y:0 }, { x: maxVal, y: maxVal }],
          type: "line",
          borderColor: "rgba(0,212,255,0.5)",
          borderWidth: 1.5,
          borderDash: [5, 3],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#5a7a99", font: { family: "Jost", size: 11 } } },
        tooltip: {
          callbacks: {
            label: ctx => ` Actual ${ctx.parsed.x} → Pred ${ctx.parsed.y}`
          }
        }
      },
      scales: {
        x: { title:{ display:true, text:"Actual AQI",    color:"#5a7a99" }, ticks:{color:"#5a7a99"}, grid:{color:"rgba(255,255,255,0.04)"}},
        y: { title:{ display:true, text:"Predicted AQI", color:"#5a7a99" }, ticks:{color:"#5a7a99"}, grid:{color:"rgba(255,255,255,0.04)"}},
      }
    }
  });

  // ── Residuals bar chart ────────────────────────────────
  if (tsResidualChart) tsResidualChart.destroy();
  const residuals = data.actual
    .map((a, i) => a != null ? parseFloat((data.predicted[i] - a).toFixed(1)) : null)
    .filter(v => v != null);
  const resLabels = data.dates.filter((_, i) => data.actual[i] != null);
  const resColors = residuals.map(v =>
    v > 0 ? "rgba(255,109,0,0.55)" : "rgba(0,200,83,0.55)"
  );

  tsResidualChart = new Chart(document.getElementById("ts-residual-chart"), {
    type: "bar",
    data: {
      labels: resLabels,
      datasets: [{
        label: "Residual (Pred − Actual)",
        data:  residuals,
        backgroundColor: resColors,
        borderWidth: 0,
        barPercentage: 0.8,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels:{ color:"#5a7a99", font:{ family:"Jost", size:11 }}},
        tooltip: { callbacks:{ label: ctx => ` ${ctx.parsed.y > 0 ? "+" : ""}${ctx.parsed.y} AQI` }}
      },
      scales: {
        x: { ticks:{ color:"#5a7a99", maxTicksLimit:10, maxRotation:0, font:{size:10} }, grid:{color:"rgba(255,255,255,0.03)"}},
        y: {
          ticks:{ color:"#5a7a99" }, grid:{ color:"rgba(255,255,255,0.04)" },
          title:{ display:true, text:"Pred − Actual", color:"#5a7a99" }
        }
      }
    }
  });
}