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
// ÉTAT GLOBAL
// ═══════════════════════════════════════════
let currentDays  = 1;
let histChart    = null;
let compareChart = null;
let dataTable    = null;
let allRecords   = [];

function setStatus(msg) {
  document.getElementById("statusMsg").textContent = msg;
  document.getElementById("lastUpdate").textContent =
    "Mise à jour : " + new Date().toLocaleTimeString("fr-MA");
}

// ═══════════════════════════════════════════
// FILTRE PÉRIODE
// ═══════════════════════════════════════════
function setDays(days, btn) {
  currentDays = days;
  document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  loadAll();
}

// ═══════════════════════════════════════════
// COULEUR COLONNE PV
// ═══════════════════════════════════════════
function pvClass(pv) {
  if (pv > 35) return "td-pv-high";
  if (pv > 10) return "td-pv-medium";
  return "td-pv-low";
}

// ═══════════════════════════════════════════
// CHART HISTORIQUE
// ═══════════════════════════════════════════
function initHistChart(labels, values) {
  const ctx = document.getElementById("histChart").getContext("2d");
  if (histChart) histChart.destroy();

  histChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Prévision PV (kW)",
        data: values,
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245,158,11,0.07)",
        borderWidth: 1.5,
        pointRadius: 0, pointHoverRadius: 4,
        fill: true, tension: 0.3
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
          callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(2)} kW` }
        }
      },
      scales: {
        x: { ticks: { color: "#475569", font: { size: 9 }, maxRotation: 45, maxTicksLimit: 20 }, grid: { color: "rgba(255,255,255,0.03)" } },
        y: { min: 0, ticks: { color: "#475569", font: { size: 10 }, callback: v => v + " kW" }, grid: { color: "rgba(255,255,255,0.04)" } }
      }
    }
  });
}

// ═══════════════════════════════════════════
// CHARGER HISTORIQUE + RÉSUMÉ + GRAPHE + TABLEAU
// ═══════════════════════════════════════════
async function loadHistory() {
  document.getElementById("chartLoadingHist").classList.remove("hidden");
  document.getElementById("detailLoading").classList.remove("hidden");

  try {
    const res  = await fetch(`/api/history?days=${currentDays}`);
    const data = await res.json();

    if (data.status !== "ok" || data.records.length === 0) {
      allRecords = [];
      document.getElementById("summaryCount").textContent  = "0";
      document.getElementById("summaryMax").textContent    = "--";
      document.getElementById("summaryMean").textContent   = "--";
      document.getElementById("summaryEnergy").textContent = "--";
      renderTable([]);
      return;
    }

    allRecords = data.records;
    const pvValues = allRecords.map(r => parseFloat(r.pv_forecast_kw) || 0);
    const pvMax    = Math.max(...pvValues);
    const pvMean   = pvValues.reduce((s,v) => s+v, 0) / pvValues.length;
    const energy   = pvValues.reduce((s,v) => s+v, 0) * 0.5;

    document.getElementById("summaryCount").textContent  = allRecords.length;
    document.getElementById("summaryMax").textContent    = pvMax.toFixed(2);
    document.getElementById("summaryMean").textContent   = pvMean.toFixed(2);
    document.getElementById("summaryEnergy").textContent = energy.toFixed(1);

    initHistChart(
      allRecords.map(r => r.timestamp).reverse(),
      pvValues.slice().reverse()
    );

    renderTable(allRecords);

  } catch(e) {
    console.error("loadHistory:", e);
  } finally {
    document.getElementById("chartLoadingHist").classList.add("hidden");
    document.getElementById("detailLoading").classList.add("hidden");
  }
}

// ═══════════════════════════════════════════
// RENDU TABLEAU AVEC DataTables
// ═══════════════════════════════════════════
function renderTable(records) {
  if (dataTable) {
    dataTable.destroy();
    document.getElementById("detailBody").innerHTML = "";
  }

  document.getElementById("detailBody").innerHTML = records.map(r => `
    <tr>
      <td>${r.timestamp}</td>
      <td class="${pvClass(parseFloat(r.pv_forecast_kw))}">${parseFloat(r.pv_forecast_kw).toFixed(2)}</td>
      <td>${r.temperature}</td>
      <td>${r.global_radiation}</td>
      <td>${r.humidity}</td>
      <td>${r.wind_speed}</td>
      <td>${r.solar_elevation}</td>
      <td><span class="sky-badge">${r.sky_label}</span></td>
    </tr>
  `).join("");

  dataTable = $('#detailTable').DataTable({
    pageLength: 25,
    lengthMenu: [10, 25, 50, 100, 250, 1000],
    order: [[0, 'desc']],
    language: {
      search: "Rechercher :",
      lengthMenu: "Afficher _MENU_ lignes",
      info: "_START_ à _END_ sur _TOTAL_ enregistrements",
      infoEmpty: "Aucun enregistrement",
      infoFiltered: "(filtré depuis _MAX_ enregistrements)",
      paginate: { previous: "‹", next: "›" },
      zeroRecords: "Aucun résultat trouvé"
    }
  });
}

// ═══════════════════════════════════════════
// FILTRES AVANCÉS — PV / Température / Radiation
// ═══════════════════════════════════════════
function applyFilters() {
  const pvMin   = parseFloat(document.getElementById("filterPvMin").value);
  const pvMax   = parseFloat(document.getElementById("filterPvMax").value);
  const tMin    = parseFloat(document.getElementById("filterTempMin").value);
  const tMax    = parseFloat(document.getElementById("filterTempMax").value);
  const radMin  = parseFloat(document.getElementById("filterRadMin").value);
  const radMax  = parseFloat(document.getElementById("filterRadMax").value);

  $.fn.dataTable.ext.search.pop();

  $.fn.dataTable.ext.search.push(function(settings, data) {
    const pv   = parseFloat(data[1]);
    const temp = parseFloat(data[2]);
    const rad  = parseFloat(data[3]);

    if (!isNaN(pvMin)  && pv   < pvMin)  return false;
    if (!isNaN(pvMax)  && pv   > pvMax)  return false;
    if (!isNaN(tMin)   && temp < tMin)   return false;
    if (!isNaN(tMax)   && temp > tMax)   return false;
    if (!isNaN(radMin) && rad  < radMin) return false;
    if (!isNaN(radMax) && rad  > radMax) return false;

    return true;
  });

  dataTable.draw();
  setStatus(`Filtre appliqué — ${dataTable.rows({ search: 'applied' }).count()} résultats`);
}

function resetFilters() {
  document.getElementById("filterPvMin").value   = "";
  document.getElementById("filterPvMax").value   = "";
  document.getElementById("filterTempMin").value = "";
  document.getElementById("filterTempMax").value = "";
  document.getElementById("filterRadMin").value  = "";
  document.getElementById("filterRadMax").value  = "";

  $.fn.dataTable.ext.search.pop();
  dataTable.search("").draw();
  setStatus("Filtres réinitialisés");
}

// ═══════════════════════════════════════════
// STATS JOURNALIÈRES
// ═══════════════════════════════════════════
async function loadStats() {
  document.getElementById("tableLoading").classList.remove("hidden");
  try {
    const res  = await fetch(`/api/history/stats?days=${currentDays}`);
    const data = await res.json();

    if (data.status !== "ok" || data.stats.length === 0) {
      document.getElementById("statsBody").innerHTML =
        `<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:2rem">Aucune donnée</td></tr>`;
      return;
    }

    document.getElementById("statsBody").innerHTML = data.stats.map(s => `
      <tr>
        <td>${s.date}</td>
        <td>${s.nb_records}</td>
        <td class="${pvClass(s.pv_max)}">${s.pv_max}</td>
        <td>${s.pv_mean}</td>
        <td style="color:var(--amber);font-weight:600">${s.pv_total_kwh} kWh</td>
      </tr>
    `).join("");

  } catch(e) {
    console.error("loadStats:", e);
  } finally {
    document.getElementById("tableLoading").classList.add("hidden");
  }
}

// ═══════════════════════════════════════════
// COMPARAISON JOUR / SEMAINE / MOIS
// ═══════════════════════════════════════════
async function loadComparison() {
  document.getElementById("compareLoading").classList.remove("hidden");

  try {
    const res  = await fetch(`/api/history?days=30`);
    const data = await res.json();

    if (data.status !== "ok" || data.records.length === 0) {
      document.getElementById("compareBody").innerHTML =
        `<tr><td colspan="4" style="text-align:center;color:var(--text3);padding:1rem">Aucune donnée</td></tr>`;
      return;
    }

    const records = data.records.map(r => ({
      ts: new Date(r.timestamp.replace(" ", "T")),
      pv: parseFloat(r.pv_forecast_kw) || 0
    }));

    const now = new Date();
    const oneDayAgo   = new Date(now - 1  * 24*60*60*1000);
    const oneWeekAgo  = new Date(now - 7  * 24*60*60*1000);
    const oneMonthAgo = new Date(now - 30 * 24*60*60*1000);

    const jour    = records.filter(r => r.ts >= oneDayAgo);
    const semaine = records.filter(r => r.ts >= oneWeekAgo);
    const mois    = records.filter(r => r.ts >= oneMonthAgo);

    function computeStats(arr) {
      if (arr.length === 0) return { energy: 0, mean: 0, max: 0 };
      const pvs = arr.map(r => r.pv);
      return {
        energy: pvs.reduce((s,v) => s+v, 0) * 0.5,
        mean:   pvs.reduce((s,v) => s+v, 0) / pvs.length,
        max:    Math.max(...pvs)
      };
    }

    const statsJour    = computeStats(jour);
    const statsSemaine = computeStats(semaine);
    const statsMois    = computeStats(mois);

    document.getElementById("compareBody").innerHTML = `
      <tr class="compare-row-jour">
        <td>● Aujourd'hui (24h)</td>
        <td class="text-end">${statsJour.energy.toFixed(1)}</td>
        <td class="text-end">${statsJour.mean.toFixed(2)}</td>
        <td class="text-end">${statsJour.max.toFixed(2)}</td>
      </tr>
      <tr class="compare-row-semaine">
        <td>● Cette semaine (7j)</td>
        <td class="text-end">${statsSemaine.energy.toFixed(1)}</td>
        <td class="text-end">${statsSemaine.mean.toFixed(2)}</td>
        <td class="text-end">${statsSemaine.max.toFixed(2)}</td>
      </tr>
      <tr class="compare-row-mois">
        <td>● Ce mois (30j)</td>
        <td class="text-end">${statsMois.energy.toFixed(1)}</td>
        <td class="text-end">${statsMois.mean.toFixed(2)}</td>
        <td class="text-end">${statsMois.max.toFixed(2)}</td>
      </tr>
    `;

    // Graphe comparatif (barres) — couleurs thème dark
    const ctx = document.getElementById("compareChart").getContext("2d");
    if (compareChart) compareChart.destroy();

    compareChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Énergie (kWh)", "PV Moyen (kW)", "PV Max (kW)"],
        datasets: [
          {
            label: "Jour (24h)",
            data: [statsJour.energy, statsJour.mean, statsJour.max],
            backgroundColor: "#f59e0b"
          },
          {
            label: "Semaine (7j)",
            data: [statsSemaine.energy, statsSemaine.mean, statsSemaine.max],
            backgroundColor: "#38bdf8"
          },
          {
            label: "Mois (30j)",
            data: [statsMois.energy, statsMois.mean, statsMois.max],
            backgroundColor: "#a78bfa"
          }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: true,
        plugins: {
          legend: { labels: { color: "#94a3b8", font: { family: "'JetBrains Mono', monospace", size: 11 } } },
          tooltip: {
            backgroundColor: "#243447", borderColor: "#334155", borderWidth: 1,
            titleColor: "#f8fafc", bodyColor: "#94a3b8", padding: 10
          }
        },
        scales: {
          x: { ticks: { color: "#94a3b8", font: { size: 11 } }, grid: { display: false } },
          y: { ticks: { color: "#475569", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" } }
        }
      }
    });

  } catch(e) {
    console.error("loadComparison:", e);
  } finally {
    document.getElementById("compareLoading").classList.add("hidden");
  }
}

// ═══════════════════════════════════════════
// EXPORT CSV (brut, depuis le serveur)
// ═══════════════════════════════════════════
function exportCSV() {
  window.open("/api/history/export", "_blank");
}

// ═══════════════════════════════════════════
// EXPORT EXCEL — SheetJS, côté navigateur, stylé
// ═══════════════════════════════════════════
function exportExcel() {
  if (!dataTable) return;

  const rows = dataTable.rows({ search: 'applied' }).data().toArray();

  if (rows.length === 0) {
    setStatus("Aucune donnée à exporter");
    return;
  }

  const header = [
    "Timestamp", "PV (kW)", "Température (°C)", "Radiation (W/m²)",
    "Humidité (%)", "Vent (km/h)", "Élévation (°)", "État ciel"
  ];

  const tmp = document.createElement("div");
  const dataRows = rows.map(r => r.map(cell => {
    tmp.innerHTML = cell;
    return tmp.textContent.trim();
  }));

  const wsData = [header, ...dataRows];
  const ws = XLSX.utils.aoa_to_sheet(wsData);

  ws['!cols'] = [
    { wch: 18 }, { wch: 10 }, { wch: 14 }, { wch: 14 },
    { wch: 12 }, { wch: 12 }, { wch: 12 }, { wch: 16 }
  ];

  const range = XLSX.utils.decode_range(ws['!ref']);
  for (let C = range.s.c; C <= range.e.c; C++) {
    const cellRef = XLSX.utils.encode_cell({ r: 0, c: C });
    if (!ws[cellRef]) continue;
    ws[cellRef].s = {
      font: { bold: true, color: { rgb: "000000" } },
      fill: { fgColor: { rgb: "FFC107" } },
      alignment: { horizontal: "center" }
    };
  }

  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Historique PV");

  const filename = `historique_pvforecast_${new Date().toISOString().slice(0,10)}.xlsx`;
  XLSX.writeFile(wb, filename, { cellStyles: true });

  setStatus(`Export Excel : ${dataRows.length} lignes`);
}

// ═══════════════════════════════════════════
// LOAD ALL
// ═══════════════════════════════════════════
async function loadAll() {
  setStatus("Chargement en cours…");
  await Promise.all([loadHistory(), loadStats(), loadComparison()]);
  setStatus("Données chargées · auto-actualisation 30min");
}

// ═══════════════════════════════════════════
// AUTO-REFRESH — toutes les 30 minutes
// Recharge les données même sans intervention
// (le CSV est lui-même alimenté par le scheduler
//  serveur côté app.py, indépendamment de cette page)
// ═══════════════════════════════════════════
setInterval(loadAll, 30 * 60 * 1000);

document.addEventListener("DOMContentLoaded", loadAll);