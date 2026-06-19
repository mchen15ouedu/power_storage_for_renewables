/* Solar + Wind Storage Atlas — interactive front-end.
   Loads web/data/{meta.json, regions.geojson, regions/<id>.json} produced by
   scripts/build_website.py and renders a Leaflet choropleth + per-region detail. */
"use strict";

const DATA = "data";
const PLOTLY_BG = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#cdd9e2", size: 10 }, margin: { l: 44, r: 12, t: 24, b: 32 } };

// per-metric color ramp: [domain low, high] filled at load from data; reverse=true
// means "high is worse" (storage) so the ramp runs cool->hot.
const RAMPS = {
  mix_s_tot_pct:  { stops: ["#1a9850", "#fee08b", "#d73027"], pct: true },
  solar_s_tot_pct:{ stops: ["#1a9850", "#fee08b", "#d73027"], pct: true },
  wind_s_tot_pct: { stops: ["#1a9850", "#fee08b", "#d73027"], pct: true },
  solar_cf:       { stops: ["#08306b", "#4292c6", "#ffd24d"] },
  wind_cf:        { stops: ["#3f007d", "#807dba", "#4cc9a0"] },
  mix_alpha:      { stops: ["#2c7fb8", "#ffffbf", "#f4a259"] }, // wind<->solar
};

let META, INDEX, GEO, layer, map, curMetric, curDataset = "gldas", domain = {};

function lerp(a, b, t) { return a + (b - a) * t; }
function hex2rgb(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
function rgb2hex(c) { return "#" + c.map(v => Math.round(v).toString(16).padStart(2, "0")).join(""); }
function rampColor(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const seg = 1 / (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(t / seg));
  const f = (t - i * seg) / seg, a = hex2rgb(stops[i]), b = hex2rgb(stops[i + 1]);
  return rgb2hex([0, 1, 2].map(k => lerp(a[k], b[k], f)));
}
function colorFor(props) {
  if (props.no_data) return "#3a444e";
  const v = props[curMetric];
  if (v === null || v === undefined) return "#39424c";
  const d = domain[curMetric], t = (v - d[0]) / (d[1] - d[0] || 1);
  return rampColor(RAMPS[curMetric].stops, t);
}
function fmt(v, dp = 2) { return v === null || v === undefined ? "—" : (+v).toFixed(dp); }
function fmtPct(v, dp = 1) { return v === null || v === undefined ? "—" : (+v).toFixed(dp) + "%"; }

async function boot() {
  META = await fetch(`${DATA}/meta.json`).then(r => r.json());
  INDEX = await fetch(`${DATA}/regions_index.json`).then(r => r.json());
  GEO = await fetch(`${DATA}/regions.geojson`).then(r => r.json());
  document.title = META.title;
  document.querySelector("#sidebar header h1").innerHTML = META.title.replace(/ \+ /, "&nbsp;+&nbsp;");
  buildMetricSelect();
  computeDomains();
  initMap();
  styleLayer();
  buildLegend();
  buildSearch();
  buildDownloads();
  document.getElementById("meta-foot").innerHTML =
    `Period ${META.period}. GLDAS theoretical forcing for the world; ` +
    `USA states add real EIA consumption. NLDAS-USA: ` +
    (META.datasets.nldas.available ? "available." : "pending (analysis not yet run).");
  document.getElementById("closeDetail").onclick = closeDetail;
  document.querySelectorAll(".ds-tab").forEach(b =>
    b.onclick = () => { if (!b.classList.contains("disabled")) selectDataset(b.dataset.ds); });
  // collapsible panels
  const app = document.getElementById("app");
  const remap = () => setTimeout(() => map && map.invalidateSize(), 230);
  document.getElementById("sidebarCollapse").onclick = () => { app.classList.add("left-collapsed"); remap(); };
  document.getElementById("sidebarReopen").onclick = () => { app.classList.remove("left-collapsed"); remap(); };
  document.getElementById("detailReopen").onclick = () => {
    document.getElementById("detail").classList.remove("hidden");
    document.getElementById("detailReopen").classList.remove("show");
  };
}

function buildMetricSelect() {
  const sel = document.getElementById("metricSelect");
  META.map_metrics.forEach(k => {
    const o = document.createElement("option");
    o.value = k; o.textContent = (META.metric_info[k] || {}).label || k;
    sel.appendChild(o);
  });
  curMetric = META.map_metrics[0];
  sel.value = curMetric;
  sel.onchange = () => { curMetric = sel.value; styleLayer(); buildLegend(); };
}

function computeDomains() {
  META.map_metrics.forEach(k => {
    const vals = GEO.features.map(f => f.properties[k]).filter(v => v !== null && v !== undefined)
      .sort((a, b) => a - b);
    if (!vals.length) { domain[k] = [0, 1]; return; }
    const q = p => vals[Math.floor(p * (vals.length - 1))];
    domain[k] = [q(0.02), q(0.98)];  // clip outliers for a readable ramp
  });
}

function initMap() {
  map = L.map("map", { worldCopyJump: true, minZoom: 2, maxZoom: 9 }).setView([26, 6], 2);
  const sat = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics", maxZoom: 19 });
  const terrain = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
    { attribution: "&copy; Esri", maxZoom: 13 });
  const topo = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    { attribution: "&copy; Esri", maxZoom: 19 });
  const dark = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
    { attribution: "&copy; OpenStreetMap &copy; CARTO", subdomains: "abcd", maxZoom: 19 });
  const labels = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    { attribution: "", maxZoom: 19, pane: "shadowPane" });
  sat.addTo(map); labels.addTo(map);
  L.control.layers(
    { "Satellite": sat, "Topographic": topo, "Terrain": terrain, "Dark": dark },
    { "Place labels": labels }, { position: "bottomleft", collapsed: true }
  ).addTo(map);
  layer = L.geoJSON(GEO, {
    style: f => baseStyle(f),
    onEachFeature: (f, lyr) => {
      lyr.on({
        mouseover: e => { e.target.setStyle({ weight: 1.4, color: "#fff", fillOpacity: .92 });
          e.target.bindTooltip(tip(f.properties), { className: "region-tip", sticky: true }).openTooltip(); },
        mouseout: e => layer.resetStyle(e.target),
        click: () => openRegion(f.properties.id),
      });
    },
  }).addTo(map);
}
function baseStyle(f) {
  return { fillColor: colorFor(f.properties), weight: .5, color: "rgba(255,255,255,.35)",
    fillOpacity: f.properties.no_data ? .12 : .62 };
}
function styleLayer() { if (layer) layer.setStyle(f => baseStyle(f)); }
function tip(p) {
  const mi = META.metric_info[curMetric] || {};
  let v = p[curMetric];
  v = v === null || v === undefined ? "no data" : (+v).toFixed(2) + (RAMPS[curMetric].pct ? "%" : "");
  return `<div class="region-tip"><b>${p.name}</b><br>${p.level === "admin1" ? p.country + " · " : ""}` +
    `${mi.label || curMetric}: ${v}${p.has_usa ? " · USA ✓" : ""}</div>`;
}

function buildLegend() {
  const el = document.getElementById("legend"), r = RAMPS[curMetric], d = domain[curMetric];
  const grad = r.stops.map((s, i) => `${s} ${(100 * i / (r.stops.length - 1)).toFixed(0)}%`).join(", ");
  const mi = META.metric_info[curMetric] || {};
  el.innerHTML =
    `<div class="bar" style="background:linear-gradient(90deg,${grad})"></div>` +
    `<div class="ends"><span>${(+d[0]).toFixed(1)}${r.pct ? "%" : ""}</span>` +
    `<span>${(+d[1]).toFixed(1)}${r.pct ? "%" : ""}</span></div>` +
    `<div style="margin-top:6px">${mi.help || ""}</div>`;
}

function buildDownloads() {
  const el = document.getElementById("dlLinks");
  if (!el) return;
  const nice = { "summary.csv": "World summary (770)", "usa_summary.csv": "USA states (51)",
    "storage_ranking_full.csv": "Storage ranking", "colocation_storage.csv": "Co-location storage",
    "usa_months_unmet.csv": "USA months unmet", "usa_storage_ranking.csv": "USA storage ranking" };
  el.innerHTML = (META.downloads || []).map(f =>
    `<a href="${DATA}/downloads/${f}" download>⤓ ${nice[f] || f}</a>`).join("");
}

function buildSearch() {
  const inp = document.getElementById("search"), out = document.getElementById("searchResults");
  inp.oninput = () => {
    const q = inp.value.trim().toLowerCase();
    out.innerHTML = "";
    if (q.length < 2) return;
    INDEX.filter(r => r.name.toLowerCase().includes(q) ||
                      (r.country || "").toLowerCase().includes(q))
      .slice(0, 30).forEach(r => {
        const li = document.createElement("li");
        li.innerHTML = `${r.name} <span class="c">${r.level === "admin1" ? r.country : "country"}` +
          `${r.has_usa ? " · USA" : ""}</span>`;
        li.onclick = () => { openRegion(r.id); out.innerHTML = ""; inp.value = r.name; };
        out.appendChild(li);
      });
  };
}

/* ---------------- detail drawer ---------------- */
let curRegion = null;
async function openRegion(id) {
  const reg = await fetch(`${DATA}/regions/${id}.json`).then(r => r.json()).catch(() => null);
  if (!reg) return;
  curRegion = reg;
  document.getElementById("rName").textContent = reg.name;
  document.getElementById("rCrumb").textContent =
    (reg.level === "admin1" ? reg.country + " · " : "") + (reg.continent || "") +
    (reg.usa ? " · real EIA consumption" : "");
  const links = [`<a href="${DATA}/regions/${id}.json" download>⤓ data (JSON)</a>`];
  if (reg.pdf) links.unshift(`<a href="${DATA}/${reg.pdf}" target="_blank">⤓ detailed figures (PDF)</a>`);
  document.getElementById("rLinks").innerHTML = links.join("");
  const tabs = document.getElementById("dsTabs");
  if (reg.usa) {
    tabs.classList.remove("hidden");
    const nl = document.querySelector('.ds-tab[data-ds="nldas"]');
    nl.classList.toggle("disabled", !META.datasets.nldas.available);
    nl.textContent = META.datasets.nldas.label + (META.datasets.nldas.available ? "" : " · pending");
  } else { tabs.classList.add("hidden"); }
  curDataset = "gldas";
  setTabActive();
  document.getElementById("detail").classList.remove("hidden");
  document.getElementById("detailReopen").classList.remove("show");
  renderDataset();
  // fly to region
  const f = GEO.features.find(x => x.properties.id === id);
  if (f) { try { map.fitBounds(L.geoJSON(f).getBounds(), { maxZoom: 5, padding: [40, 40] }); } catch (e) {} }
}
function closeDetail() {
  document.getElementById("detail").classList.add("hidden");
  if (curRegion) document.getElementById("detailReopen").classList.add("show");
}
function selectDataset(ds) { curDataset = ds; setTabActive(); renderDataset(); }
function setTabActive() {
  document.querySelectorAll(".ds-tab").forEach(b => b.classList.toggle("active", b.dataset.ds === curDataset));
}

function renderDataset() {
  const reg = curRegion, body = document.getElementById("detailBody");
  if (curDataset === "nldas") {
    if (!META.datasets.nldas.available || !reg.datasets.nldas) {
      body.innerHTML = `<div class="pending">NLDAS-USA results (hourly forcing + EIA demand)
        are not generated yet — the reduction/analysis hasn't completed. This panel will
        populate automatically once <code>nldas_analysis</code> is built.</div>`;
      return;
    }
  }
  const ds = reg.datasets[curDataset] || {};
  // dataset-specific blocks: NLDAS tab uses its own real/metrics, GLDAS uses reg's
  const metrics = (curDataset === "nldas" && ds.metrics) ? ds.metrics : reg.metrics;
  const usaB = reg.usa ? (curDataset === "nldas" && ds.real ? ds.real : reg.usa) : null;
  let html = numberCards(metrics);
  if (usaB) html += usaSection(usaB);
  if (!ds.series_norm) {
    html += `<div class="pending" style="margin-top:14px">Per-region charts for this region
      haven't been generated yet (run the builder with <code>--scope world</code>). The numbers
      above are complete; detailed multi-year curves are in the bundled PDFs.</div>`;
    body.innerHTML = html;
    return;
  }
  html += `<div class="section-h">Deficit plots — supply vs demand (normalized)</div>
           <div id="chartSolar" class="chart mini"></div>
           <div id="chartWind" class="chart mini"></div>
           <div id="chartMix" class="chart mini"></div>
           <div class="chart-cap">Monthly climatology, mean-normalized: each resource's supply
           (after worst-year overbuild) vs demand = 1.0 (dotted). Below the dotted line = a deficit
           drawn from storage; above = surplus that recharges it.</div>`;
  if (ds.series_norm && ds.series_norm.mix_storage)
    html += `<div class="section-h">Storage state through the year</div>
             <div id="chartStore" class="chart"></div>
             <div class="chart-cap">Mean storage level by month (fraction of the sized capacity);
             the seasonal draw-down that sets the storage requirement.</div>`;
  if (reg.usa && ds.series_real)
    html += `<div class="section-h">Real energy supply vs EIA demand (TWh)</div>
             <div id="chartReal" class="chart"></div>
             <div class="chart-cap">Optimal-mix monthly generation (TWh) sized to meet demand,
             shown unconstrained and scaled to fit each land cap. Lines below the black demand
             curve are an energy deficit storage cannot fix.</div>`;
  if (reg.colocation) html += colocationSection(reg);
  html += assumptionsBox();
  body.innerHTML = html;
  drawCharts(ds, reg);
  if (reg.colocation) drawColocation(reg.colocation);
  wireAssumptions();
}

function colocationSection(reg) {
  const c = reg.colocation, caps = META.land_caps;
  let grid = "";
  if (c.usa_feasible) {
    const tierColors = { "1pct": "#1a9850", oilgas2x: "#fdae61", oilgas_all: "#66bd63" };
    grid = `<table class="kmatrix"><tr><th>land cap \\ overlap</th>` +
      c.k.map(k => `<th>k=${k}%</th>`).join("") + `</tr>` +
      caps.map(cp => `<tr><td><span class="dot" style="background:${tierColors[cp.key]}"></span>${cp.label.split(" ")[0]}</td>` +
        c.usa_feasible[cp.key].map(ok => `<td class="${ok ? "yc" : "nc"}">${ok ? "✓" : "·"}</td>`).join("") +
        `</tr>`).join("") + `</table>`;
  }
  return `<div class="section-h">Co-location overlap (k)</div>
    <div id="chartColoc" class="chart" style="height:200px"></div>
    <div class="chart-cap">Storage need falls as the overlap k lets one footprint host more
    generation; the winning technology for the extra build is labeled.</div>${grid}`;
}

function assumptionsBox() {
  const a = META.assumptions || {};
  const rows = Object.keys(a).map(k => `<div class="arow"><b>${k}.</b> ${a[k]}</div>`).join("");
  return `<details class="assumptions"><summary>Assumptions behind these numbers</summary>
    <div class="abody">${rows}</div></details>`;
}
function wireAssumptions() { /* native <details>, nothing to wire */ }

function numberCards(m) {
  const mi = META.metric_info;
  const card = (k, val, unit) => `<div class="card"><div class="k">${(mi[k] || {}).label || k}</div>
    <div class="v">${val}<small>${unit || (mi[k] || {}).unit || ""}</small></div></div>`;
  return `<div class="section-h">Key numbers (normalized)</div><div class="cards">
    ${card("solar_cf", fmt(m.solar_cf, 3))}
    ${card("wind_cf", fmt(m.wind_cf, 3))}
    ${card("mix_alpha", fmt(m.mix_alpha, 2), " solar")}
    ${card("mix_s_tot_pct", fmtPct(m.mix_s_tot_pct))}
    ${card("solar_s_tot_pct", fmtPct(m.solar_s_tot_pct))}
    ${card("wind_s_tot_pct", fmtPct(m.wind_s_tot_pct))}
    ${card("mix_f_adj", fmt(m.mix_f_adj, 2), "×")}
    ${card("wind_flaute_days", fmt(m.wind_flaute_days, 0), " d")}
  </div>`;
}

function usaSection(u) {
  const caps = META.land_caps;
  const tierColors = { "1pct": "#1a9850", oilgas2x: "#fdae61", oilgas_all: "#66bd63" };
  const card = (k, v, u2) => `<div class="card"><div class="k">${k}</div>
    <div class="v">${v}<small>${u2 || ""}</small></div></div>`;
  let tiers = `<div class="tiers">`;
  caps.forEach(c => {
    const info = u.caps[c.key] || {};
    const ok = info.feasible === true;
    tiers += `<div class="tier"><span class="dot" style="background:${tierColors[c.key] || "#888"}"></span>
      <span class="lbl">${c.label}</span>
      <span class="val">${ok ? "storage " + fmt(info.storage_TWh, 1) + " TWh"
        : "short " + fmt(info.shortfall_TWh, 1) + " TWh"}</span>
      <span class="badge ${ok ? "yes" : "no"}">${ok ? "feasible" : "over cap"}</span></div>`;
  });
  tiers += `</div>`;
  return `<div class="section-h">Real units · EIA consumption</div><div class="cards">
    ${card("Annual demand", fmt(u.annual_consumption_TWh, 1), " TWh/yr")}
    ${card("Mix capacity", fmt(u.mix_capacity_TWh, 1), " TWh-rated")}
    ${card("Solar nameplate", fmt(u.mix_solar_nameplate_GW, 1), " GW")}
    ${card("Wind nameplate", fmt(u.mix_wind_nameplate_GW, 1), " GW")}
    ${card("Land (optimal mix)", fmtPct(u.mix_land_pct), " of state")}
    ${card("Solar-only land", fmtPct(u.solar_land_pct), " of state")}
  </div>
  <div class="section-h">Feasibility by land cap</div>${tiers}`;
}

/* ---------------- charts ---------------- */
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
function deficitPlot(divId, supply, demand, name, color, fillc) {
  if (!document.getElementById(divId)) return;
  Plotly.newPlot(divId, [
    { x: MONTHS, y: supply, name, mode: "lines", line: { color, width: 2.2 },
      fill: "tozeroy", fillcolor: fillc },
    { x: MONTHS, y: demand, name: "demand", mode: "lines",
      line: { color: "#e6edf3", width: 1.4, dash: "dot" } },
  ], Object.assign({ height: 168, showlegend: false,
    title: { text: name, font: { size: 12, color: color }, x: 0.04, y: 0.93 },
    margin: { l: 38, r: 10, t: 6, b: 22 },
    yaxis: { title: "", zeroline: false, rangemode: "tozero" },
    xaxis: { tickfont: { size: 9 } } }, PLOTLY_BG),
    { displayModeBar: false, responsive: true });
}

function drawCharts(ds, reg) {
  if (ds.series_norm) {
    const s = ds.series_norm;
    deficitPlot("chartSolar", s.solar_supply, s.demand, "Solar", "#f4a259", "rgba(244,162,89,.14)");
    deficitPlot("chartWind", s.wind_supply, s.demand, "Wind", "#4cc9a0", "rgba(76,201,160,.14)");
    deficitPlot("chartMix", s.mix_supply, s.demand, "Optimal mix", "#5b9bd5", "rgba(91,155,213,.15)");

    if (s.mix_storage)
      Plotly.newPlot("chartStore", [{ x: MONTHS, y: s.mix_storage, mode: "lines",
        fill: "tozeroy", line: { color: "#d73027", width: 2 }, fillcolor: "rgba(215,48,39,.15)" }],
        Object.assign({ height: 240, yaxis: { title: "storage (frac. of capacity)" } }, PLOTLY_BG),
        { displayModeBar: false, responsive: true });
  }
  if (reg.usa && ds.series_real) {
    const r = ds.series_real;
    const capColor = { nocap: "#1a9850", oilgas_all: "#74add1", oilgas2x: "#fdae61", "1pct": "#d73027" };
    const capName = { nocap: "no cap", oilgas_all: "≤5% land", oilgas2x: "≤1.88%", "1pct": "≤1% land" };
    const traces = Object.keys(capColor).filter(k => r["supply_TWh_" + k]).map(k =>
      ({ x: MONTHS, y: r["supply_TWh_" + k], name: capName[k], mode: "lines",
         line: { color: capColor[k], width: 2 } }));
    traces.push({ x: MONTHS, y: r.demand_TWh, name: "EIA demand", mode: "lines",
      line: { color: "#cdd9e2", width: 1.6, dash: "dot" } });
    Plotly.newPlot("chartReal", traces, Object.assign({ height: 240, showlegend: true,
      legend: { orientation: "h", y: -0.2, font: { size: 9 } },
      yaxis: { title: "TWh / month" } }, PLOTLY_BG), { displayModeBar: false, responsive: true });
  }
}

function drawColocation(c) {
  const pickColor = { base: "#888", solar: "#f4a259", wind: "#4cc9a0" };
  const xs = c.k.map(k => k + "%");
  Plotly.newPlot("chartColoc", [{
    x: xs, y: c.storage_pct_cons, mode: "lines+markers+text",
    text: c.pick.map(p => p || ""), textposition: "top center",
    textfont: { size: 9, color: "#9fb0bf" },
    line: { color: "#5b9bd5", width: 2 },
    marker: { size: 9, color: c.pick.map(p => pickColor[p] || "#888") },
  }], Object.assign({ height: 200,
    xaxis: { title: "overlap k" }, yaxis: { title: "storage (% annual cons.)" } }, PLOTLY_BG),
    { displayModeBar: false, responsive: true });
}

boot();
