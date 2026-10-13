// ========================================================
// PureAir — Map-First Navigation & Hyperlocal AQI Engine
// ========================================================

// Default coordinate (Bengaluru MG Road)
const DEFAULT_COORDS = {
  lat: 12.9756,
  lon: 77.6066,
  name: "Mahatma Gandhi Road, Bengaluru",
};

// State
let appMode = "route"; // "route" or "street"
let routeState = "ready"; // "picking-a" | "picking-b" | "ready"
let pointA = { lat: 12.9756, lon: 77.6066, name: "MG Road, Bengaluru" };
let pointB = { lat: 12.9352, lon: 77.6245, name: "Koramangala 4th Block, Bengaluru" };
let inspectPoint = { ...DEFAULT_COORDS };
let isFetchingRoute = false;

// Map Layers
let markerA = null;
let markerB = null;
let inspectMarker = null;
let shortestRouteLayer = null;
let cleanestRouteLayer = null;

// DOM Elements - Nav
const navModeRoute = document.querySelector("#nav-mode-route");
const navModeStreet = document.querySelector("#nav-mode-street");
const drawerViewRoute = document.querySelector("#drawer-view-route");
const drawerViewStreet = document.querySelector("#drawer-view-street");
const cityButtons = document.querySelectorAll(".city-btn");

// DOM Elements - On-Map Helper & Instruction Pill
const mapInstructionPill = document.querySelector("#map-instruction-pill");
const mapInstructionText = document.querySelector("#map-instruction-text");
const btnResetMapPoints = document.querySelector("#btn-reset-map-points");

// DOM Elements - Point A & B Cards
const cardPointA = document.querySelector("#card-point-a");
const cardPointB = document.querySelector("#card-point-b");
const textPointA = document.querySelector("#text-point-a");
const textPointB = document.querySelector("#text-point-b");
const btnReselectA = document.querySelector("#btn-reselect-a");
const btnReselectB = document.querySelector("#btn-reselect-b");
const btnSwapPoints = document.querySelector("#btn-swap-points");
const inputRouteSearch = document.querySelector("#input-route-search");
const btnRouteSearchGo = document.querySelector("#btn-route-search-go");
const btnRunRouting = document.querySelector("#btn-run-routing");
const btnRunRoutingText = document.querySelector("#btn-run-routing-text");
const routeErrorMsg = document.querySelector("#route-error-msg");

// DOM Elements - Route Results HUD
const routeResultsCard = document.querySelector("#route-results-card");
const advPct = document.querySelector("#adv-pct");
const advSummary = document.querySelector("#adv-summary");
const resCleanestDist = document.querySelector("#res-cleanest-dist");
const resCleanestTime = document.querySelector("#res-cleanest-time");
const resCleanestPm = document.querySelector("#res-cleanest-pm");
const resCleanestAqi = document.querySelector("#res-cleanest-aqi");
const resCleanestNote = document.querySelector("#res-cleanest-note");
const resShortestDist = document.querySelector("#res-shortest-dist");
const resShortestTime = document.querySelector("#res-shortest-time");
const resShortestPm = document.querySelector("#res-shortest-pm");
const resShortestAqi = document.querySelector("#res-shortest-aqi");
const resShortestNote = document.querySelector("#res-shortest-note");

// DOM Elements - Street AQI Explorer
const streetSearchForm = document.querySelector("#street-search-form");
const streetSearchInput = document.querySelector("#street-search-input");
const meterRoadName = document.querySelector("#meter-road-name");
const meterCategoryBadge = document.querySelector("#meter-category-badge");
const meterAqiNum = document.querySelector("#meter-aqi-num");
const meterStandard = document.querySelector("#meter-standard");
const streetMicroIcon = document.querySelector("#street-micro-icon");
const streetMicroTitle = document.querySelector("#street-micro-title");
const streetMicroDesc = document.querySelector("#street-micro-desc");
const valStreetPm = document.querySelector("#val-street-pm");
const valBasePm = document.querySelector("#val-base-pm");
const valTrafficPct = document.querySelector("#val-traffic-pct");
const val24hPm = document.querySelector("#val-24h-pm");

// ========================================================
// 1. INITIALIZE HIGH-END MAP (OpenStreetMap - No API Key Needed)
// ========================================================
const map = L.map("map", {
  zoomControl: false,
  scrollWheelZoom: true,
}).setView([pointA.lat, pointA.lon], 13);

// Reposition Zoom Control to bottom-right
L.control.zoom({ position: "bottomright" }).addTo(map);

// Free OpenStreetMap Tiles (Removes "API KEY REQUIRED" watermark)
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

// ========================================================
// 2. CUSTOM SVG PIN BUILDERS
// ========================================================
function createPinIcon(letter, isStart) {
  const bubbleClass = isStart ? "pin-bubble-a" : "pin-bubble-b";
  return L.divIcon({
    className: "custom-map-pin",
    html: `
      <div class="pin-anchor-wrap">
        <div class="pin-bubble ${bubbleClass}">
          <span>${letter}</span>
        </div>
        <div class="pin-radar-ring"></div>
      </div>
    `,
    iconSize: [36, 44],
    iconAnchor: [18, 44],
    popupAnchor: [0, -44],
  });
}

const pinIconA = createPinIcon("A", true);
const pinIconB = createPinIcon("B", false);

const inspectPinIcon = L.divIcon({
  className: "custom-map-pin",
  html: '<div class="inspect-dot-pin"></div>',
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

// ========================================================
// 3. REVERSE GEOCODING HELPER
// ========================================================
async function reverseGeocode(lat, lon) {
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=jsonv2&zoom=18`
    );
    if (res.ok) {
      const data = await res.json();
      const addr = data.address || {};
      const road = addr.road || addr.suburb || addr.neighbourhood || data.name || "Street Location";
      const area = addr.suburb || addr.city_district || addr.city || "";
      return area ? `${road}, ${area}` : road;
    }
  } catch {}
  return `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
}

// ========================================================
// 4. MAP PIN PLACEMENT & DRAGGING (Points A & B)
// ========================================================
function renderPinA(lat, lon, label) {
  if (markerA) map.removeLayer(markerA);
  markerA = L.marker([lat, lon], {
    icon: pinIconA,
    draggable: true,
  }).addTo(map);

  markerA.bindPopup(`<b>Start Point (A)</b><br>${label || "Origin"}`);

  // When dragged directly on map, recalculate routes!
  markerA.on("dragend", async (e) => {
    const pos = markerA.getLatLng();
    pointA.lat = pos.lat;
    pointA.lon = pos.lng;
    textPointA.textContent = "Updating address…";
    const name = await reverseGeocode(pos.lat, pos.lng);
    pointA.name = name;
    textPointA.textContent = name;
    markerA.setPopupContent(`<b>Start Point (A)</b><br>${name}`);
    if (pointB) computeEcoRoutes();
  });
}

function renderPinB(lat, lon, label) {
  if (markerB) map.removeLayer(markerB);
  markerB = L.marker([lat, lon], {
    icon: pinIconB,
    draggable: true,
  }).addTo(map);

  markerB.bindPopup(`<b>Destination (B)</b><br>${label || "Destination"}`);

  // When dragged directly on map, recalculate routes!
  markerB.on("dragend", async (e) => {
    const pos = markerB.getLatLng();
    pointB.lat = pos.lat;
    pointB.lon = pos.lng;
    textPointB.textContent = "Updating address…";
    const name = await reverseGeocode(pos.lat, pos.lng);
    pointB.name = name;
    textPointB.textContent = name;
    markerB.setPopupContent(`<b>Destination (B)</b><br>${name}`);
    if (pointA) computeEcoRoutes();
  });
}

// Set up initial pins A and B
renderPinA(pointA.lat, pointA.lon, pointA.name);
renderPinB(pointB.lat, pointB.lon, pointB.name);
textPointA.textContent = pointA.name;
textPointB.textContent = pointB.name;

// ========================================================
// 5. INTERACTIVE MAP CLICK STATE MACHINE
// ========================================================
map.on("click", async (e) => {
  const lat = e.latlng.lat;
  const lon = e.latlng.lng;

  // In Street AQI Mode
  if (appMode === "street") {
    inspectPoint = { lat, lon, name: "Selected Location" };
    renderInspectMarker(lat, lon);
    fetchStreetAQI(lat, lon);
    return;
  }

  // In Route Mode: user locates A or B directly on map
  if (routeState === "picking-a") {
    pointA = { lat, lon, name: "Locating…" };
    textPointA.textContent = "Locating…";
    renderPinA(lat, lon, "Start Point (A)");
    const name = await reverseGeocode(lat, lon);
    pointA.name = name;
    textPointA.textContent = name;
    markerA.setPopupContent(`<b>Start Point (A)</b><br>${name}`);

    // If B doesn't exist, advance to picking B
    if (!pointB) {
      setRoutePickingState("picking-b");
    } else {
      setRoutePickingState("ready");
      computeEcoRoutes();
    }
  } else if (routeState === "picking-b") {
    pointB = { lat, lon, name: "Locating…" };
    textPointB.textContent = "Locating…";
    renderPinB(lat, lon, "Destination (B)");
    const name = await reverseGeocode(lat, lon);
    pointB.name = name;
    textPointB.textContent = name;
    markerB.setPopupContent(`<b>Destination (B)</b><br>${name}`);

    setRoutePickingState("ready");
    computeEcoRoutes();
  } else {
    // If already ready, clicking map repositions B (or whichever is selected)
    pointB = { lat, lon, name: "Locating…" };
    textPointB.textContent = "Locating…";
    renderPinB(lat, lon, "Destination (B)");
    const name = await reverseGeocode(lat, lon);
    pointB.name = name;
    textPointB.textContent = name;
    markerB.setPopupContent(`<b>Destination (B)</b><br>${name}`);
    computeEcoRoutes();
  }
});

function setRoutePickingState(state) {
  routeState = state;
  cardPointA.classList.remove("active");
  cardPointB.classList.remove("active");

  if (state === "picking-a") {
    cardPointA.classList.add("active");
    mapInstructionText.innerHTML = "Click anywhere on the map to place <strong>Start Point (A)</strong>";
    btnRunRoutingText.textContent = "Click map to set Start Point (A)…";
  } else if (state === "picking-b") {
    cardPointB.classList.add("active");
    mapInstructionText.innerHTML = "Click anywhere on the map to place <strong>Destination Point (B)</strong>";
    btnRunRoutingText.textContent = "Click map to set Destination Point (B)…";
  } else {
    mapInstructionText.innerHTML = "✨ Drag pins <strong>A</strong> or <strong>B</strong> anywhere on the map to update routes!";
    btnRunRoutingText.textContent = "Recalculate Cleanest vs Shortest Route";
  }
}

// Button "Locate on Map" for Point A
btnReselectA.addEventListener("click", (e) => {
  e.stopPropagation();
  setRoutePickingState("picking-a");
});

// Button "Locate on Map" for Point B
btnReselectB.addEventListener("click", (e) => {
  e.stopPropagation();
  setRoutePickingState("picking-b");
});

// Reset Map Points Button
btnResetMapPoints.addEventListener("click", () => {
  if (shortestRouteLayer) map.removeLayer(shortestRouteLayer);
  if (cleanestRouteLayer) map.removeLayer(cleanestRouteLayer);
  if (markerA) map.removeLayer(markerA);
  if (markerB) map.removeLayer(markerB);
  markerA = null;
  markerB = null;
  pointA = null;
  pointB = null;
  textPointA.textContent = "Click on map to place pin…";
  textPointB.textContent = "Click on map to place pin…";
  routeResultsCard.hidden = true;
  setRoutePickingState("picking-a");
});

// Swap Points A & B
btnSwapPoints.addEventListener("click", () => {
  if (!pointA || !pointB) return;
  const temp = { ...pointA };
  pointA = { ...pointB };
  pointB = { ...temp };

  textPointA.textContent = pointA.name;
  textPointB.textContent = pointB.name;
  renderPinA(pointA.lat, pointA.lon, pointA.name);
  renderPinB(pointB.lat, pointB.lon, pointB.name);
  computeEcoRoutes();
});

// ========================================================
// 6. COMPUTE ECO-ROUTES (Cleanest Green vs Shortest Red)
// ========================================================
async function computeEcoRoutes() {
  if (!pointA || !pointB) {
    routeErrorMsg.textContent = "Please place both Start (A) and Destination (B) on the map.";
    return;
  }
  if (isFetchingRoute) return;
  isFetchingRoute = true;
  routeErrorMsg.textContent = "";
  btnRunRouting.disabled = true;
  btnRunRoutingText.textContent = "Calculating Cleanest vs Shortest Route…";

  try {
    const query = new URLSearchParams({
      origin_lat: String(pointA.lat),
      origin_lon: String(pointA.lon),
      dest_lat: String(pointB.lat),
      dest_lon: String(pointB.lon),
      origin_name: pointA.name,
      dest_name: pointB.name,
    });

    const res = await fetch(`/cleanest_route?${query}`);
    const data = await res.json();

    if (!res.ok || !data.available) {
      throw new Error(data.detail || "Could not generate routes for these coordinates.");
    }

    renderRoutesOnMap(data);
    displayRouteResultsHUD(data);
    setRoutePickingState("ready");
  } catch (err) {
    routeErrorMsg.textContent = err.message || "Failed to calculate routes.";
  } finally {
    isFetchingRoute = false;
    btnRunRouting.disabled = false;
    btnRunRoutingText.textContent = "Recalculate Cleanest vs Shortest Route";
  }
}

btnRunRouting.addEventListener("click", computeEcoRoutes);

// Optional: Type search for address
btnRouteSearchGo.addEventListener("click", async () => {
  const q = inputRouteSearch.value.trim();
  if (!q) return;
  try {
    const res = await fetch(`/search_place?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    if (res.ok && data.lat && data.lon) {
      const lat = parseFloat(data.lat);
      const lon = parseFloat(data.lon);
      const name = data.name || q;
      // If A is waiting, set A; else set B
      if (routeState === "picking-a" || !pointA) {
        pointA = { lat, lon, name };
        textPointA.textContent = name;
        renderPinA(lat, lon, name);
        map.setView([lat, lon], 14);
        setRoutePickingState("picking-b");
      } else {
        pointB = { lat, lon, name };
        textPointB.textContent = name;
        renderPinB(lat, lon, name);
        map.setView([lat, lon], 14);
        computeEcoRoutes();
      }
    }
  } catch {}
});

// Render Polylines on Leaflet Map
function renderRoutesOnMap(data) {
  if (shortestRouteLayer) map.removeLayer(shortestRouteLayer);
  if (cleanestRouteLayer) map.removeLayer(cleanestRouteLayer);

  const shortest = data.shortest_route;
  const cleanest = data.cleanest_route;
  const comparison = data.comparison;

  // 1. Draw Shortest Route (RED)
  shortestRouteLayer = L.polyline(shortest.polyline, {
    color: "#ef4444",
    weight: 5,
    opacity: 0.85,
    lineCap: "round",
    lineJoin: "round",
  }).addTo(map);

  shortestRouteLayer.bindPopup(`
    <div style="font-family:sans-serif; min-width: 190px;">
      <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
        <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#ef4444;"></span>
        <strong style="color:#b91c1c; font-size:14px;">Shortest Route (Red)</strong>
      </div>
      <p style="margin:0; font-size:12px; color:#334155; line-height: 1.4;">
        Distance: <b>${shortest.distance_km} km</b> · Time: <b>${shortest.duration_min} min</b><br>
        Avg PM2.5: <b style="color:#dc2626;">${shortest.avg_pm25} µg/m³</b> (AQI ${shortest.aqi})<br>
        <span style="color:#b91c1c; font-size:11px; font-weight:600;">⚠️ Higher vehicular exhaust exposure</span>
      </p>
    </div>
  `);

  // 2. Draw Cleanest Route (EMERALD GREEN)
  cleanestRouteLayer = L.polyline(cleanest.polyline, {
    color: "#10b981",
    weight: 6.5,
    opacity: 0.95,
    lineCap: "round",
    lineJoin: "round",
  }).addTo(map);

  cleanestRouteLayer.bindPopup(`
    <div style="font-family:sans-serif; min-width: 200px;">
      <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
        <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#10b981;"></span>
        <strong style="color:#047857; font-size:14px;">Cleanest Route (Green)</strong>
      </div>
      <p style="margin:0; font-size:12px; color:#334155; line-height: 1.4;">
        Distance: <b>${cleanest.distance_km} km</b> · Time: <b>${cleanest.duration_min} min</b><br>
        Avg PM2.5: <b style="color:#059669;">${cleanest.avg_pm25} µg/m³</b> (AQI ${cleanest.aqi})<br>
        <span style="color:#047857; font-weight:bold; font-size:12px;">🍃 ${comparison.exposure_reduction_pct}% Cleaner Air!</span>
      </p>
    </div>
  `);

  // Auto-fit bounds to show both routes gracefully
  const group = new L.featureGroup([shortestRouteLayer, cleanestRouteLayer]);
  map.fitBounds(group.getBounds().pad(0.15));
}

// Display Route HUD Card
function displayRouteResultsHUD(data) {
  routeResultsCard.hidden = false;
  const shortest = data.shortest_route;
  const cleanest = data.cleanest_route;
  const comparison = data.comparison;

  advPct.textContent = `${comparison.exposure_reduction_pct}% LESS EXPOSURE`;
  advSummary.textContent = comparison.summary;

  // Cleanest (Green)
  resCleanestDist.textContent = `${cleanest.distance_km} km`;
  resCleanestTime.textContent = `${cleanest.duration_min} min`;
  resCleanestPm.textContent = `${cleanest.avg_pm25} µg/m³`;
  resCleanestAqi.textContent = `${cleanest.aqi} (${cleanest.aqi_category})`;
  resCleanestNote.textContent = cleanest.highlights;

  // Shortest (Red)
  resShortestDist.textContent = `${shortest.distance_km} km`;
  resShortestTime.textContent = `${shortest.duration_min} min`;
  resShortestPm.textContent = `${shortest.avg_pm25} µg/m³`;
  resShortestAqi.textContent = `${shortest.aqi} (${shortest.aqi_category})`;
  resShortestNote.textContent = shortest.highlights;
}

// ========================================================
// 7. STREET AQI EXPLORER LOGIC
// ========================================================
function renderInspectMarker(lat, lon) {
  if (inspectMarker) map.removeLayer(inspectMarker);
  inspectMarker = L.marker([lat, lon], {
    icon: inspectPinIcon,
  }).addTo(map);
}

async function fetchStreetAQI(lat, lon) {
  meterRoadName.textContent = "Loading road data…";
  meterAqiNum.textContent = "…";
  valStreetPm.textContent = "…";
  valBasePm.textContent = "…";
  valTrafficPct.textContent = "…";
  val24hPm.textContent = "…";

  try {
    const query = new URLSearchParams({ lat: String(lat), lon: String(lon) });
    const res = await fetch(`/air_quality?${query}`);
    const data = await res.json();

    if (!res.ok || data.available === false) {
      meterRoadName.textContent = "Data unavailable";
      meterAqiNum.textContent = "—";
      return;
    }

    const streetAqi = data.street_aqi !== undefined ? data.street_aqi : Math.round(data.aqi);
    const streetPm = data.street_pm2_5 !== undefined ? data.street_pm2_5 : data.pm2_5;
    const basePm = data.baseline_pm2_5 !== undefined ? data.baseline_pm2_5 : data.pm2_5;
    const trafficFactor = data.traffic_factor_pct !== undefined ? data.traffic_factor_pct : 0;
    const roadName = data.road_name || "Local Road";
    const microDesc = data.microenvironment || "Urban Corridor";
    const aqiCat = data.aqi_category || "Moderate";
    const aqiCol = data.aqi_color || "#10b981";

    meterRoadName.textContent = roadName;
    meterCategoryBadge.textContent = aqiCat;
    meterCategoryBadge.style.color = aqiCol;
    meterAqiNum.textContent = streetAqi;
    meterAqiNum.style.color = aqiCol;
    meterStandard.textContent = data.aqi_standard || "CPCB Standard";

    const isPark = data.road_type === "park" || data.road_type === "garden";
    const isTrunk = data.road_type === "motorway" || data.road_type === "trunk";
    streetMicroIcon.textContent = isPark ? "🌳" : isTrunk ? "⚡" : "🚗";
    streetMicroTitle.textContent = `${roadName} (${data.road_type || "Road"})`;
    streetMicroDesc.textContent = microDesc;

    valStreetPm.textContent = `${streetPm.toFixed(1)} µg/m³`;
    valBasePm.textContent = `${basePm.toFixed(1)} µg/m³`;
    valTrafficPct.textContent = `${trafficFactor > 0 ? "+" : ""}${trafficFactor}%`;
    valTrafficPct.className = `breakdown-val ${trafficFactor > 15 ? "text-red" : "text-green"}`;
    val24hPm.textContent = Number.isFinite(data.pm2_5_24h) ? `${data.pm2_5_24h.toFixed(1)} µg/m³` : "Model baseline";

    inspectMarker.bindPopup(`
      <div style="font-family:sans-serif; min-width: 170px;">
        <strong style="color:#0f172a; font-size:14px;">${roadName}</strong><br>
        <span style="color:${aqiCol}; font-weight:800; font-size:16px;">AQI ${streetAqi} (${aqiCat})</span><br>
        <span style="font-size:12px; color:#475569;">PM2.5: <b>${streetPm.toFixed(1)} µg/m³</b></span><br>
        <small style="color:#64748b;">${microDesc}</small>
      </div>
    `).openPopup();
  } catch {
    meterRoadName.textContent = "Error fetching air quality";
  }
}

streetSearchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = streetSearchInput.value.trim();
  if (!q) return;
  try {
    const res = await fetch(`/search_place?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    if (res.ok && data.lat && data.lon) {
      const lat = parseFloat(data.lat);
      const lon = parseFloat(data.lon);
      const name = data.name || q;
      // If A is waiting, set A; else set B
      if (routeState === "picking-a" || !pointA) {
        pointA = { lat, lon, name };
        textPointA.textContent = name;
        renderPinA(lat, lon, name);
        map.setView([lat, lon], 14);
        setRoutePickingState("picking-b");
      } else {
        pointB = { lat, lon, name };
        textPointB.textContent = name;
        renderPinB(lat, lon, name);
        map.setView([lat, lon], 14);
        computeEcoRoutes();
      }
    }
  } catch {}
});

// ========================================================
// 8. MODE SWITCHER & CITY JUMPERS
// ========================================================
navModeRoute.addEventListener("click", () => {
  appMode = "route";
  navModeRoute.classList.add("active");
  navModeRoute.setAttribute("aria-selected", "true");
  navModeStreet.classList.remove("active");
  navModeStreet.setAttribute("aria-selected", "false");

  drawerViewRoute.classList.add("active");
  drawerViewRoute.hidden = false;
  drawerViewStreet.classList.remove("active");
  drawerViewStreet.hidden = true;

  mapInstructionPill.style.display = "flex";
  if (inspectMarker) map.removeLayer(inspectMarker);
  if (pointA && pointB) computeEcoRoutes();
});

navModeStreet.addEventListener("click", () => {
  appMode = "street";
  navModeStreet.classList.add("active");
  navModeStreet.setAttribute("aria-selected", "true");
  navModeRoute.classList.remove("active");
  navModeRoute.setAttribute("aria-selected", "false");

  drawerViewStreet.classList.add("active");
  drawerViewStreet.hidden = false;
  drawerViewRoute.classList.remove("active");
  drawerViewRoute.hidden = true;

  mapInstructionPill.style.display = "none";
  renderInspectMarker(inspectPoint.lat, inspectPoint.lon);
  fetchStreetAQI(inspectPoint.lat, inspectPoint.lon);
});

// City Jumper buttons
cityButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    cityButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const lat = parseFloat(btn.dataset.lat);
    const lon = parseFloat(btn.dataset.lon);
    const name = btn.dataset.name;

    map.setView([lat, lon], 13);

    // Setup coordinates for new city
    pointA = { lat: lat + 0.005, lon: lon - 0.005, name: `Central ${name}` };
    pointB = { lat: lat - 0.015, lon: lon + 0.012, name: `Outer ${name}` };
    renderPinA(pointA.lat, pointA.lon, pointA.name);
    renderPinB(pointB.lat, pointB.lon, pointB.name);
    textPointA.textContent = pointA.name;
    textPointB.textContent = pointB.name;

    inspectPoint = { lat, lon, name };
    if (appMode === "route") {
      computeEcoRoutes();
    } else {
      renderInspectMarker(lat, lon);
      fetchStreetAQI(lat, lon);
    }
  });
});

// ========================================================
// 9. INITIAL LOAD
// ========================================================
computeEcoRoutes();