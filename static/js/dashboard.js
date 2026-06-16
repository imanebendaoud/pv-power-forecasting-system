// ═══════════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════════
function updateClock() {
  document.getElementById("clock").textContent =
    new Date().toLocaleTimeString("fr-MA", { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════
// CHART PV
// ═══════════════════════════════════════════
let forecastChart = null;

function initChart(labels, data) {
  const ctx = document.getElementById("forecastChart").getContext("2d");
  if (forecastChart) forecastChart.destroy();

  const maxIdx = data.indexOf(Math.max(...data));

  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Prévision PV (kW)",
        data,
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245,158,11,0.08)",
        borderWidth: 3,
        pointBackgroundColor: data.map((_,i) => i===maxIdx ? "#fbbf24" : "#f59e0b"),
        pointRadius: data.map((_,i) => i===maxIdx ? 6 : 3),
        pointHoverRadius: 7,
        pointBorderWidth: 2,
        pointBorderColor: "#17212b",
        fill: true, tension: 0.4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { family: "'JetBrains Mono', monospace", size: 11 } } },
        tooltip: {
          backgroundColor: "#243447", borderColor: "#334155", borderWidth: 1,
          titleColor: "#f59e0b", bodyColor: "#f8fafc", padding: 10,
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(2)} kW`,
            afterLabel: ctx => ctx.dataIndex === maxIdx ? " ◀ pic" : ""
          }
        }
      },
      scales: {
        x: { ticks: { color: "#475569", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.03)" } },
        y: { min: 0, ticks: { color: "#475569", font: { size: 10 }, callback: v => v + " kW" }, grid: { color: "rgba(255,255,255,0.04)" } }
      }
    }
  });
}

// ═══════════════════════════════════════════
// CHART MÉTÉO
// ═══════════════════════════════════════════
let meteoChart = null;

function initMeteoChart(labels, radiations, temperatures) {
  const ctx = document.getElementById("meteoChart").getContext("2d");
  if (meteoChart) meteoChart.destroy();

  meteoChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Radiation (W/m²)",
          data: radiations,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56,189,248,0.07)",
          borderWidth: 2, pointRadius: 2, pointHoverRadius: 5,
          fill: true, tension: 0.4, yAxisID: "yRad"
        },
        {
          label: "Température (°C)",
          data: temperatures,
          borderColor: "#f87171",
          backgroundColor: "transparent",
          borderWidth: 2, pointRadius: 2, pointHoverRadius: 5,
          fill: false, tension: 0.4, yAxisID: "yTemp"
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { family: "'JetBrains Mono', monospace", size: 11 }, usePointStyle: true } },
        tooltip: { backgroundColor: "#243447", borderColor: "#334155", borderWidth: 1, titleColor: "#f8fafc", bodyColor: "#94a3b8", padding: 10 }
      },
      scales: {
        x:    { ticks: { color: "#475569", font: { size: 10 }, maxRotation: 0 }, grid: { color: "rgba(255,255,255,0.03)" } },
        yRad: { position: "left",  min: 0, ticks: { color: "#38bdf8", font: { size: 10 }, callback: v => v + " W/m²" }, grid: { color: "rgba(255,255,255,0.04)" } },
        yTemp:{ position: "right",         ticks: { color: "#f87171", font: { size: 10 }, callback: v => v + "°C"    }, grid: { display: false } }
      }
    }
  });
}

// ═══════════════════════════════════════════
// STATUS
// ═══════════════════════════════════════════
function setStatus(type, msg) {
  document.getElementById("statusDot").className = "status-dot " + type;
  document.getElementById("statusMsg").textContent = msg;
}
function setLastUpdate() {
  document.getElementById("lastUpdate").textContent =
    "Dernière mise à jour : " + new Date().toLocaleTimeString("fr-MA");
}

// ═══════════════════════════════════════════
// ÉTAT DU CIEL
// ═══════════════════════════════════════════
function updateSkyCard(sky) {
  const card = document.getElementById("cardSky");
  card.className = "card card-sky sky-" + sky.level;
  document.getElementById("skyIcon").textContent  = sky.icon;
  document.getElementById("skyState").textContent = sky.label;
}

// ═══════════════════════════════════════════
// NIVEAU — couleur dynamique
// ═══════════════════════════════════════════
function updateNiveauCard(pct) {
  const card  = document.getElementById("miniKpiNiveau");
  const label = document.getElementById("prodLevel");
  card.className = "mini-kpi";
  if (pct <= 0)       { label.textContent = "Arrêt"; card.classList.add("level-faible"); }
  else if (pct <= 40) { label.textContent = "Faible"; card.classList.add("level-faible"); }
  else if (pct <= 60) { label.textContent = "Moyen";  card.classList.add("level-moyen");  }
  else if (pct <= 80) { label.textContent = "Élevé";  card.classList.add("level-eleve");  }
  else                { label.textContent = "Fort";   card.classList.add("level-fort");   }
}

// ═══════════════════════════════════════════
// DATE DU JOUR
// ═══════════════════════════════════════════
function todayLabel() {
  return new Date().toLocaleDateString("fr-MA", {
    weekday: "long", day: "numeric", month: "long"
  });
}

// ═══════════════════════════════════════════
// FETCH — PRÉVISION NOW
// ═══════════════════════════════════════════
async function fetchNow() {
  try {
    const res  = await fetch("/api/forecast/now");
    const data = await res.json();
    if (data.status !== "ok") {
      document.getElementById("pvValue").textContent = "Err";
      return;
    }
    const pv   = data.pv_kw;
    const time = data.timestamp.split(" ")[1];
    document.getElementById("pvValue").textContent = pv.toFixed(2);
    document.getElementById("pvTime").textContent  = time;
    const pct = Math.min((pv / 58) * 100, 100);
    document.getElementById("pvBar").style.width      = pct + "%";
    document.getElementById("utilRate").textContent   = pct.toFixed(1) + "%";
    document.getElementById("plantState").textContent = pv <= 0 ? "Arrêt" : "Actif";
    updateNiveauCard(pct);
  } catch(e) {
    document.getElementById("pvValue").textContent = "Err";
    console.error("fetchNow:", e);
  }
}

// ═══════════════════════════════════════════
// FETCH — MÉTÉO + ÉTAT DU CIEL
// ═══════════════════════════════════════════
async function fetchWeather() {
  try {
    const res  = await fetch("/api/weather");
    const data = await res.json();
    if (data.status !== "ok") return;
    document.getElementById("meteoTemp").textContent    = data.temperature;
    document.getElementById("meteoHum").textContent     = data.humidity;
    document.getElementById("meteoWind").textContent    = data.wind_speed;
    document.getElementById("meteoClear").textContent   = data.clearness_index;
    document.getElementById("meteoModTemp").textContent = data.module_temperature;
    document.getElementById("statElev").textContent     = data.solar_elevation + "°";
    document.getElementById("statRad").textContent      = data.global_radiation;
    const badge = document.getElementById("dayBadge");
    badge.textContent = data.is_day ? "☀ JOUR" : "☾ NUIT";
    badge.className   = "day-badge " + (data.is_day ? "jour" : "nuit");
    updateSkyCard(data.sky);
  } catch(e) {
    console.error("fetchWeather:", e);
  }
}

// ═══════════════════════════════════════════
// FETCH — COURBE PV + INDICATEURS
// ═══════════════════════════════════════════
async function fetchChart() {
  document.getElementById("chartLoading").classList.remove("hidden");
  try {
    const res  = await fetch("/api/forecast/12h");
    const data = await res.json();
    if (data.status !== "ok") return;
    const labels = data.points.map(p => p.time);
    const values = data.points.map(p => p.pv_kw);
    initChart(labels, values);
    const energy  = values.reduce((s, v) => s + v, 0) * 0.5;
    const maxPv   = Math.max(...values);
    const maxTime = labels[values.indexOf(maxPv)];
    document.getElementById("indicEnergy").textContent   = energy.toFixed(1);
    document.getElementById("indicPeak").textContent     = maxPv.toFixed(2);
    document.getElementById("indicPeakTime").textContent = maxTime;
    document.getElementById("indicDate").textContent     = todayLabel();
    setStatus("ok", `Pic prévu : ${maxPv.toFixed(1)} kW à ${maxTime}`);
  } catch(e) {
    console.error("fetchChart:", e);
  } finally {
    document.getElementById("chartLoading").classList.add("hidden");
  }
}

// ═══════════════════════════════════════════
// FETCH — GRAPHE MÉTÉO 12H
// ═══════════════════════════════════════════
async function fetchMeteoChart() {
  document.getElementById("chartLoadingMeteo").classList.remove("hidden");
  try {
    const res  = await fetch("/api/weather/12h");
    const data = await res.json();
    if (data.status !== "ok") return;
    const labels = data.chart.map(r => r.time);
    const rads   = data.chart.map(r => r.radiation);
    const temps  = data.chart.map(r => r.temperature);
    initMeteoChart(labels, rads, temps);
  } catch(e) {
    console.error("fetchMeteoChart:", e);
  } finally {
    document.getElementById("chartLoadingMeteo").classList.add("hidden");
  }
}

// ═══════════════════════════════════════════
// REFRESH ALL
// ═══════════════════════════════════════════
async function refreshAll() {
  const btn = document.getElementById("refreshBtn");
  btn.classList.add("spinning");
  setStatus("loading", "Mise à jour en cours…");
  await Promise.allSettled([fetchNow(), fetchWeather()]);
  fetchChart();
  fetchMeteoChart();
  setLastUpdate();
  btn.classList.remove("spinning");
}

setInterval(refreshAll, 30 * 60 * 1000);
document.addEventListener("DOMContentLoaded", refreshAll);