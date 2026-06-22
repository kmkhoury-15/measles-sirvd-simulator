/*
  Age-Structured Measles SEIR/SIRVD Simulator
  Browser-only JavaScript port of the uploaded Python model.
  No backend or build step is required.
*/

const AGE_LABELS = ["0–4 yrs", "5–14 yrs", "15–44 yrs", "45+ yrs"];
const N_AGES = 4;
const AGE_COLORS = ["#E8624C", "#F5A623", "#4C9BE8", "#4CE87A"];

// Approximate POLYMOD-derived contact matrix used by the Python script.
// Rows = age of individual, columns = age of contact, units = contacts/day.
const C_DAILY = [
  [8.5, 1.0, 3.5, 0.5],
  [3.0, 8.0, 3.0, 0.5],
  [6.5, 7.5, 7.0, 1.5],
  [3.5, 5.5, 2.0, 3.0]
];
const C_YEARLY = C_DAILY.map(row => row.map(value => value * 365));

const DEFAULTS = {
  N_total: 12000,
  t_duration: 120,
  t_unit: "days",
  R0_target: 12,
  alpha_season: 0.2,
  incubation_days: 12,
  infectious_days: 9,
  CFR: 0.02,
  mu: 0.018,
  I0_count: 5,
  seed_age: 0,
  efficacy1: 0.93,
  efficacy2: 0.97,
  age_fracs: "0.08, 0.15, 0.45, 0.32",
  prior_immune_frac: "0.05, 0.20, 0.40, 0.50",
  nu1: "0.01, 0.01, 0.02, 0.01",
  nu2: "0, 0.08, 0.01, 0.005"
};

const chartState = {};
let latestResults = null;

function $(id) {
  return document.getElementById(id);
}

function formatInt(value) {
  if (!Number.isFinite(value)) return "—";
  return Math.round(value).toLocaleString();
}

function formatPct(value) {
  if (!Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function parseVector(id, expectedLength, label) {
  const raw = $(id).value.trim();
  const vector = raw.split(",").map(item => Number(item.trim()));
  if (vector.length !== expectedLength || vector.some(value => !Number.isFinite(value))) {
    throw new Error(`${label} must contain ${expectedLength} comma-separated numeric values.`);
  }
  return vector;
}

function readNumeric(id, { min = -Infinity, max = Infinity } = {}) {
  const value = Number($(id).value);
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${id} must be a number between ${min} and ${max}.`);
  }
  return value;
}

function readParams() {
  const params = {
    N_total: readNumeric("N_total", { min: 100 }),
    t_duration: readNumeric("t_duration", { min: 1 }),
    t_unit: $("t_unit").value,
    R0_target: readNumeric("R0_target", { min: 1, max: 30 }),
    alpha_season: readNumeric("alpha_season", { min: 0, max: 1 }),
    incubation_days: readNumeric("incubation_days", { min: 1, max: 30 }),
    infectious_days: readNumeric("infectious_days", { min: 1, max: 30 }),
    CFR: readNumeric("CFR", { min: 0, max: 0.5 }),
    mu: readNumeric("mu", { min: 0, max: 0.1 }),
    I0_count: readNumeric("I0_count", { min: 1 }),
    seed_age: Number($("seed_age").value),
    efficacy1: readNumeric("efficacy1", { min: 0, max: 1 }),
    efficacy2: readNumeric("efficacy2", { min: 0, max: 1 }),
    age_fracs: parseVector("age_fracs", N_AGES, "Age fractions"),
    prior_immune_frac: parseVector("prior_immune_frac", N_AGES, "Prior immune fractions"),
    nu1: parseVector("nu1", N_AGES, "Dose 1 vaccination rates"),
    nu2: parseVector("nu2", N_AGES, "Dose 2 vaccination rates")
  };

  const ageSum = params.age_fracs.reduce((a, b) => a + b, 0);
  if (Math.abs(ageSum - 1) > 1e-6) {
    throw new Error("Age fractions must sum to 1.0.");
  }
  if (params.prior_immune_frac.some(value => value < 0 || value > 1)) {
    throw new Error("Prior immune fractions must be between 0 and 1.");
  }
  if (params.nu1.some(value => value < 0) || params.nu2.some(value => value < 0)) {
    throw new Error("Vaccination rates cannot be negative.");
  }
  return params;
}

function initializeDerivedParams(params) {
  const delta = (1 / params.incubation_days) * 365;
  const gamma = (1 / params.infectious_days) * 365;
  const muD = params.CFR === 0 ? 0 : (params.CFR / (1 - params.CFR)) * gamma;
  const sigma1 = 1 - params.efficacy1;
  const sigma2 = 1 - params.efficacy2;

  let tYears;
  let displayScale;
  if (params.t_unit === "years") {
    tYears = params.t_duration;
    displayScale = 1;
  } else if (params.t_unit === "months") {
    tYears = params.t_duration / 12;
    displayScale = 12;
  } else {
    tYears = params.t_duration / 365;
    displayScale = 365;
  }

  // The original Python model uses 4 time points per day, minimum 500 points.
  // A conservative cap keeps browser performance reasonable for very long runs.
  const nSteps = Math.min(Math.max(Math.floor(tYears * 365 * 4), 500), 20000);
  const tInternal = linspace(0, tYears, nSteps);
  const tDisplay = tInternal.map(t => t * displayScale);

  const NAge = params.age_fracs.map(frac => params.N_total * frac);
  const R0Age = NAge.map((n, i) => n * params.prior_immune_frac[i]);
  const S0Age = NAge.map((n, i) => n * (1 - params.prior_immune_frac[i]));
  const E0Age = Array(N_AGES).fill(0);
  const I0Age = Array(N_AGES).fill(0);
  const V10Age = Array(N_AGES).fill(0);
  const V20Age = Array(N_AGES).fill(0);
  const D0Age = Array(N_AGES).fill(0);

  const seeded = Math.min(params.I0_count, S0Age[params.seed_age]);
  I0Age[params.seed_age] = seeded;
  S0Age[params.seed_age] -= seeded;

  return {
    ...params,
    delta,
    gamma,
    muD,
    sigma1,
    sigma2,
    tYears,
    displayScale,
    nSteps,
    tInternal,
    tDisplay,
    NAge,
    S0Age,
    E0Age,
    I0Age,
    R0Age,
    V10Age,
    V20Age,
    D0Age
  };
}

function linspace(start, stop, count) {
  if (count === 1) return [start];
  const step = (stop - start) / (count - 1);
  return Array.from({ length: count }, (_, index) => start + index * step);
}

function pack(S, E, I, R, V1, V2, D) {
  return [...S, ...E, ...I, ...R, ...V1, ...V2, ...D];
}

function unpack(y) {
  return {
    S: y.slice(0, N_AGES),
    E: y.slice(N_AGES, 2 * N_AGES),
    I: y.slice(2 * N_AGES, 3 * N_AGES),
    R: y.slice(3 * N_AGES, 4 * N_AGES),
    V1: y.slice(4 * N_AGES, 5 * N_AGES),
    V2: y.slice(5 * N_AGES, 6 * N_AGES),
    D: y.slice(6 * N_AGES, 7 * N_AGES)
  };
}

function matVec(matrix, vector) {
  return matrix.map(row => row.reduce((sum, value, index) => sum + value * vector[index], 0));
}

function spectralRadiusNonNegative(matrix) {
  let vector = Array(N_AGES).fill(1 / N_AGES);
  let lambda = 0;

  for (let iter = 0; iter < 100; iter += 1) {
    const product = matVec(matrix, vector);
    const norm = product.reduce((sum, value) => sum + Math.abs(value), 0);
    if (norm === 0) return 0;
    vector = product.map(value => value / norm);
    lambda = norm;
  }
  return lambda;
}

function computeR0(beta0Val, p) {
  const exitI = p.gamma + p.muD + p.mu;
  const matrix = Array.from({ length: N_AGES }, (_, a) =>
    Array.from({ length: N_AGES }, (_, b) => {
      const susceptibleFraction = p.NAge[a] > 0 ? p.S0Age[a] / p.NAge[a] : 0;
      return beta0Val * susceptibleFraction * C_DAILY[a][b] / exitI;
    })
  );
  return spectralRadiusNonNegative(matrix);
}

function deriveBeta0(p) {
  let lo = 0.001;
  let hi = 2000;
  for (let iter = 0; iter < 60; iter += 1) {
    const mid = (lo + hi) / 2;
    if (computeR0(mid, p) < p.R0_target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (lo + hi) / 2;
}

function derivatives(y, t, p) {
  const { S, E, I, R, V1, V2 } = unpack(y);
  const betaT = p.beta0 * (1 + p.alpha_season * Math.cos(2 * Math.PI * t));
  const NLive = S.map((_, a) => S[a] + E[a] + I[a] + R[a] + V1[a] + V2[a]);
  const IOverN = I.map((value, a) => (NLive[a] > 0 ? value / NLive[a] : 0));
  const lambda = matVec(C_YEARLY, IOverN).map(value => betaT * value);

  const birthRate = p.mu * NLive.reduce((sum, value) => sum + value, 0);
  const births = Array(N_AGES).fill(0);
  births[0] = birthRate;

  const dS = Array(N_AGES);
  const dE = Array(N_AGES);
  const dI = Array(N_AGES);
  const dR = Array(N_AGES);
  const dV1 = Array(N_AGES);
  const dV2 = Array(N_AGES);
  const dD = Array(N_AGES);

  for (let a = 0; a < N_AGES; a += 1) {
    dS[a] = births[a] - lambda[a] * S[a] - p.nu1[a] * S[a] - p.mu * S[a];
    dE[a] = lambda[a] * S[a] + lambda[a] * p.sigma1 * V1[a] + lambda[a] * p.sigma2 * V2[a] - p.delta * E[a] - p.mu * E[a];
    dI[a] = p.delta * E[a] - p.gamma * I[a] - p.muD * I[a] - p.mu * I[a];
    dR[a] = p.gamma * I[a] - p.mu * R[a];
    dV1[a] = p.nu1[a] * S[a] - p.nu2[a] * V1[a] - p.mu * V1[a];
    dV2[a] = p.nu2[a] * V1[a] - p.mu * V2[a];
    dD[a] = p.muD * I[a];
  }

  return pack(dS, dE, dI, dR, dV1, dV2, dD);
}

function addScaled(y, dy, scale) {
  return y.map((value, index) => value + dy[index] * scale);
}

function rk4Step(y, t, h, p) {
  const k1 = derivatives(y, t, p);
  const k2 = derivatives(addScaled(y, k1, h / 2), t + h / 2, p);
  const k3 = derivatives(addScaled(y, k2, h / 2), t + h / 2, p);
  const k4 = derivatives(addScaled(y, k3, h), t + h, p);
  return y.map((value, index) => {
    const updated = value + (h / 6) * (k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index]);
    // Clamp tiny negative numerical noise, but avoid using clipping as a substitute for solver stability.
    return updated < 0 && updated > -1e-7 ? 0 : updated;
  });
}

function advanceWithSubsteps(y, t, outputStepYears, p) {
  // The Python version uses scipy.odeint, which adapts internally for stiff, fast measles dynamics.
  // In the browser, we use small RK4 substeps between plotted output points to keep the model stable.
  const maxInternalStepYears = 0.002 / 365; // about 2.9 minutes
  const subSteps = Math.max(1, Math.ceil(Math.abs(outputStepYears) / maxInternalStepYears));
  const h = outputStepYears / subSteps;
  let current = y;
  let currentT = t;
  for (let s = 0; s < subSteps; s += 1) {
    current = rk4Step(current, currentT, h, p);
    currentT += h;
  }
  return current;
}

function emptyAgeSeries() {
  return Array.from({ length: N_AGES }, () => []);
}

function simulate(params) {
  const p = initializeDerivedParams(params);
  p.beta0 = deriveBeta0(p);
  p.R0Check = computeR0(p.beta0, p);

  let y = pack(p.S0Age, p.E0Age, p.I0Age, p.R0Age, p.V10Age, p.V20Age, p.D0Age);
  const h = p.tInternal.length > 1 ? p.tInternal[1] - p.tInternal[0] : p.tYears;

  const series = {
    S: emptyAgeSeries(),
    E: emptyAgeSeries(),
    I: emptyAgeSeries(),
    R: emptyAgeSeries(),
    V1: emptyAgeSeries(),
    V2: emptyAgeSeries(),
    D: emptyAgeSeries()
  };

  for (let k = 0; k < p.nSteps; k += 1) {
    const state = unpack(y);
    for (const key of Object.keys(series)) {
      for (let a = 0; a < N_AGES; a += 1) {
        series[key][a].push(state[key][a]);
      }
    }
    if (k < p.nSteps - 1) {
      y = advanceWithSubsteps(y, p.tInternal[k], h, p);
    }
  }

  const totals = {};
  for (const key of Object.keys(series)) {
    totals[key] = sumAgeSeries(series[key]);
  }
  totals.NLive = totals.S.map((_, k) => totals.S[k] + totals.E[k] + totals.I[k] + totals.R[k] + totals.V1[k] + totals.V2[k]);

  const rt = computeRtSeries(p, series, totals.NLive);
  const betaSeries = p.tInternal.map(t => p.beta0 * (1 + p.alpha_season * Math.cos(2 * Math.PI * t)));

  const dtYr = p.tInternal.length > 1 ? p.tInternal[1] - p.tInternal[0] : 1;
  const incidenceTotal = totals.E.map(value => p.delta * value * dtYr * 365);
  const smoothWindow = Math.max(Math.floor(30 / 365 / dtYr), 1);
  const incidenceSmoothed = movingAverage(incidenceTotal, smoothWindow);
  const incidenceAgeSmoothed = series.E.map(ageSeries => movingAverage(ageSeries.map(value => p.delta * value * dtYr * 365), smoothWindow));

  const peakI = Math.max(...totals.I);
  const peakIndex = totals.I.indexOf(peakI);
  const totalDeaths = totals.D[totals.D.length - 1];
  const totalVaccinated = totals.V1[totals.V1.length - 1] + totals.V2[totals.V2.length - 1];

  return {
    params: p,
    series,
    totals,
    rt,
    betaSeries,
    incidenceSmoothed,
    incidenceAgeSmoothed,
    summary: {
      peakI,
      peakTime: p.tDisplay[peakIndex],
      totalDeaths,
      totalVaccinated,
      finalLiving: totals.NLive[totals.NLive.length - 1],
      R0Check: p.R0Check
    }
  };
}

function sumAgeSeries(ageSeries) {
  const length = ageSeries[0].length;
  return Array.from({ length }, (_, k) => ageSeries.reduce((sum, series) => sum + series[k], 0));
}

function computeRtSeries(p, series) {
  const exitI = p.gamma + p.muD + p.mu;
  return p.tInternal.map((t, k) => {
    const betaT = p.beta0 * (1 + p.alpha_season * Math.cos(2 * Math.PI * t));
    const matrix = Array.from({ length: N_AGES }, (_, a) => {
      const NLiveAge = series.S[a][k] + series.E[a][k] + series.I[a][k] + series.R[a][k] + series.V1[a][k] + series.V2[a][k];
      const susceptibleFraction = NLiveAge > 0 ? series.S[a][k] / Math.max(NLiveAge, 1) : 0;
      return Array.from({ length: N_AGES }, (_, b) => betaT * susceptibleFraction * C_DAILY[a][b] / exitI);
    });
    return Math.max(spectralRadiusNonNegative(matrix), 0);
  });
}

function movingAverage(values, windowSize) {
  if (windowSize <= 1) return values.slice();
  const result = Array(values.length).fill(0);
  let running = 0;
  for (let i = 0; i < values.length; i += 1) {
    running += values[i];
    if (i >= windowSize) running -= values[i - windowSize];
    const divisor = Math.min(i + 1, windowSize);
    result[i] = running / divisor;
  }
  return result;
}

function sampleIndices(length, maxPoints = 1000) {
  if (length <= maxPoints) return Array.from({ length }, (_, i) => i);
  const step = (length - 1) / (maxPoints - 1);
  return Array.from({ length: maxPoints }, (_, i) => Math.round(i * step));
}

function setStatus(message, isError = false) {
  const status = $("status");
  status.textContent = message;
  status.classList.toggle("status--error", isError);
}

function updateSummary(results) {
  const { summary, params } = results;
  $("peakInfected").textContent = `${formatInt(summary.peakI)} (${formatPct(summary.peakI / params.N_total)})`;
  $("peakTime").textContent = `${summary.peakTime.toFixed(2)} ${params.t_unit}`;
  $("totalDeaths").textContent = `${formatInt(summary.totalDeaths)} (${formatPct(summary.totalDeaths / params.N_total)})`;
  $("totalVaccinated").textContent = `${formatInt(summary.totalVaccinated)} (${formatPct(summary.totalVaccinated / params.N_total)})`;
  $("r0Check").textContent = summary.R0Check.toFixed(3);
  $("finalPopulation").textContent = formatInt(summary.finalLiving);
}

function destroyChart(id) {
  if (chartState[id]) {
    chartState[id].destroy();
    delete chartState[id];
  }
}

function chartOptions(title, yTitle, xTitle) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      title: { display: true, text: title, font: { weight: "bold" } },
      legend: { labels: { boxWidth: 14, usePointStyle: true } },
      tooltip: {
        callbacks: {
          label: context => {
            const label = context.dataset.label || "";
            const value = context.parsed.y;
            return `${label}: ${Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}`;
          }
        }
      }
    },
    scales: {
      x: { title: { display: true, text: xTitle }, ticks: { maxTicksLimit: 8 } },
      y: { title: { display: true, text: yTitle }, ticks: { maxTicksLimit: 7 } }
    }
  };
}

function makeDataset(label, data, color, extra = {}) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: `${color}22`,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.18,
    ...extra
  };
}

function renderCharts(results) {
  const p = results.params;
  const indices = sampleIndices(p.tDisplay.length);
  const labels = indices.map(i => Number(p.tDisplay[i].toFixed(2)));
  const select = values => indices.map(i => values[i]);
  const xTitle = `Time (${p.t_unit})`;

  const chartDefinitions = [
    {
      id: "chartCompartments",
      type: "line",
      title: "Total population — SEIR compartments",
      yTitle: "Individuals",
      datasets: [
        makeDataset("Susceptible (S)", select(results.totals.S), "#4C9BE8"),
        makeDataset("Exposed (E)", select(results.totals.E), "#F5A623"),
        makeDataset("Infectious (I)", select(results.totals.I), "#E8624C"),
        makeDataset("Recovered (R)", select(results.totals.R), "#4CE87A"),
        makeDataset("Living N", select(results.totals.NLive), "#7A8290", { borderDash: [4, 4], borderWidth: 1.4 })
      ]
    },
    {
      id: "chartInfectiousAge",
      type: "line",
      title: "Infectious individuals by age group",
      yTitle: "Infectious individuals",
      datasets: AGE_LABELS.map((label, a) => makeDataset(label, select(results.series.I[a]), AGE_COLORS[a]))
    },
    {
      id: "chartVaccinationDeaths",
      type: "line",
      title: "Vaccination coverage and cumulative deaths",
      yTitle: "Individuals",
      datasets: [
        makeDataset("Vaccinated dose 1 (V1)", select(results.totals.V1), "#8B5CF6"),
        makeDataset("Vaccinated dose 2 (V2)", select(results.totals.V2), "#5C3D8B"),
        makeDataset("Cumulative deaths", select(results.totals.D), "#333333", { borderDash: [5, 5] })
      ]
    },
    {
      id: "chartSusceptibleFraction",
      type: "line",
      title: "Susceptible fraction by age group (%)",
      yTitle: "% of age group susceptible",
      datasets: AGE_LABELS.map((label, a) => makeDataset(label, select(results.series.S[a].map(value => (value / p.NAge[a]) * 100)), AGE_COLORS[a]))
    },
    {
      id: "chartRt",
      type: "line",
      title: "Effective reproductive number Rt over time",
      yTitle: "Rt",
      datasets: [
        makeDataset("Rt", select(results.rt), "#8B5CF6"),
        makeDataset("Rt = 1", labels.map(() => 1), "#111827", { borderDash: [6, 6], borderWidth: 1.4 })
      ]
    },
    {
      id: "chartBeta",
      type: "line",
      title: "Seasonal forcing on transmission rate β(t)",
      yTitle: "Transmission rate β(t) (/yr)",
      datasets: [
        makeDataset("β(t) seasonal", select(results.betaSeries), "#E8624C"),
        makeDataset(`β₀ = ${p.beta0.toFixed(3)}`, labels.map(() => p.beta0), "#7A8290", { borderDash: [6, 6], borderWidth: 1.4 })
      ]
    },
    {
      id: "chartIncidence",
      type: "line",
      title: "Annualized incidence rate over time — 30-day smoothed",
      yTitle: "New infections / year",
      datasets: [
        makeDataset("Total incidence", select(results.incidenceSmoothed), "#E8624C", { fill: true, borderWidth: 2.2 }),
        ...AGE_LABELS.map((label, a) => makeDataset(label, select(results.incidenceAgeSmoothed[a]), AGE_COLORS[a], { borderDash: [4, 4], borderWidth: 1.5 }))
      ]
    }
  ];

  for (const definition of chartDefinitions) {
    destroyChart(definition.id);
    chartState[definition.id] = new Chart($(definition.id), {
      type: definition.type,
      data: { labels, datasets: definition.datasets },
      options: chartOptions(definition.title, definition.yTitle, xTitle)
    });
  }
}

function runSimulation() {
  try {
    setStatus("Running simulation...");
    const params = readParams();
    const started = performance.now();
    latestResults = simulate(params);
    updateSummary(latestResults);
    renderCharts(latestResults);
    const elapsed = ((performance.now() - started) / 1000).toFixed(2);
    setStatus(`Simulation complete in ${elapsed}s. Charts use a sampled view for speed; CSV includes the full time series.`);
  } catch (error) {
    console.error(error);
    setStatus(error.message, true);
  }
}

function resetDefaults() {
  for (const [key, value] of Object.entries(DEFAULTS)) {
    const el = $(key);
    if (el) el.value = value;
  }
  runSimulation();
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function downloadCsv() {
  if (!latestResults) {
    setStatus("Run a simulation before downloading CSV output.", true);
    return;
  }

  const r = latestResults;
  const header = [
    "time",
    "S_total",
    "E_total",
    "I_total",
    "R_total",
    "V1_total",
    "V2_total",
    "D_total",
    "N_live_total",
    "Rt",
    "beta_t",
    "incidence_30day_smoothed",
    ...AGE_LABELS.map(label => `I_${label}`),
    ...AGE_LABELS.map(label => `S_fraction_${label}`)
  ];

  const rows = [header];
  for (let k = 0; k < r.params.tDisplay.length; k += 1) {
    rows.push([
      r.params.tDisplay[k],
      r.totals.S[k],
      r.totals.E[k],
      r.totals.I[k],
      r.totals.R[k],
      r.totals.V1[k],
      r.totals.V2[k],
      r.totals.D[k],
      r.totals.NLive[k],
      r.rt[k],
      r.betaSeries[k],
      r.incidenceSmoothed[k],
      ...r.series.I.map(series => series[k]),
      ...r.series.S.map((series, a) => series[k] / r.params.NAge[a])
    ]);
  }

  const csv = rows.map(row => row.map(csvEscape).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "sirvd_measles_simulation_output.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setStatus("CSV downloaded.");
}

function initializeApp() {
  $("runButton").addEventListener("click", runSimulation);
  $("resetButton").addEventListener("click", resetDefaults);
  $("downloadButton").addEventListener("click", downloadCsv);
  runSimulation();
}

window.addEventListener("DOMContentLoaded", initializeApp);
