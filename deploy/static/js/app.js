/* ─── State ─────────────────────────────────────────────── */
const state = {
  rawPage: [], rawTotal: 0, rawPages: 0, rawCurrent: 1,
  invoices: [], invPage: 1,
  priority: [], priPage: 1,
  predicted: false,
  PAGE_SIZE: 12,
  analytics: null,
  charts: {},           // holds Chart instances so we can destroy/recreate
};

/* ─── Formatters ─────────────────────────────────────────── */
const fmt_usd = v =>
  v == null ? '—' : '$' + parseFloat(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const fmt_usd_short = v => {
  if (v == null) return '—';
  const n = parseFloat(v);
  if (Math.abs(n) >= 1_000_000) return (n < 0 ? '-' : '') + '$' + (Math.abs(n) / 1_000_000).toFixed(2) + 'M';
  if (Math.abs(n) >= 1_000)    return (n < 0 ? '-' : '') + '$' + (Math.abs(n) / 1_000).toFixed(1) + 'K';
  return (n < 0 ? '-$' : '$') + Math.abs(n).toFixed(0);
};

const fmt_num = v =>
  v == null ? '—' : parseInt(v).toLocaleString('en-US');

const fmt_date = str => {
  if (!str) return '—';
  const [y, m, d] = str.split('-');
  const mo = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  return `${mo[+m - 1]} ${+d}, ${y}`;
};

const fmt_pct = v => v == null ? '—' : parseFloat(v).toFixed(1) + '%';

/* ─── Dynamic page size ──────────────────────────────────── */
function calcPageSize() {
  const wrap = document.getElementById('invoice-tbody')?.closest('.table-wrap');
  if (!wrap || wrap.clientHeight < 60) return 12;
  const THEAD_H = 30;
  const ROW_H   = 36;
  return Math.max(5, Math.floor((wrap.clientHeight - THEAD_H) / ROW_H));
}

/* ─── Badge & cell helpers ───────────────────────────────── */
function actionBadge(action) {
  const cls = { CALL: 'badge-call', REMIND: 'badge-remind', WATCH: 'badge-watch', OK: 'badge-ok' };
  return `<span class="badge ${cls[action] || 'badge-ok'}">${action}</span>`;
}

function delayCell(days) {
  if (days == null) return '<span class="pred-date-pending">—</span>';
  const d = parseInt(days);
  let cls = 'days-ok';
  if (d > 7)       cls = 'days-high';
  else if (d >= 3) cls = 'days-mid';
  else if (d >= 1) cls = 'days-low';
  const sign = d > 0 ? '+' : '';
  return `<span class="${cls}">${sign}${d}d</span>`;
}

function probCell(p) {
  if (p == null) return '—';
  const cls = p >= 0.7 ? 'prob-high' : p >= 0.4 ? 'prob-mid' : 'prob-low';
  return `<span class="${cls}">${(p * 100).toFixed(0)}%</span>`;
}

/* ════════════════════════════════════════════════════════════
   TAB SWITCHING
   ════════════════════════════════════════════════════════════ */
function switchTab(tabName) {
  const panels = ['operations', 'impact', 'intelligence'];
  panels.forEach(id => {
    const panel = document.getElementById(`tab-${id}`);
    const btn   = document.querySelector(`[onclick="switchTab('${id}')"]`);
    if (id === tabName) {
      panel.style.display = id === 'operations' ? 'grid' : 'flex';
      btn.classList.add('active');
    } else {
      panel.style.display = 'none';
      btn.classList.remove('active');
    }
  });
}

/* ════════════════════════════════════════════════════════════
   PRE-PREDICTION: invoice table (server-side pagination)
   ════════════════════════════════════════════════════════════ */
async function fetchRawPage(page) {
  const res  = await fetch(`/api/invoices?page=${page}&limit=${state.PAGE_SIZE}`);
  const data = await res.json();
  state.rawPage    = data.invoices;
  state.rawTotal   = data.total;
  state.rawPages   = data.pages;
  state.rawCurrent = data.page;
  renderRawInvoiceTable();
  renderPagination('pagination', state.rawCurrent, state.rawPages, state.rawTotal, 'fetchRawPage');
}

function renderRawInvoiceTable() {
  document.getElementById('invoice-tbody').innerHTML =
    state.rawPage.length === 0
      ? '<tr><td colspan="6" class="loading-cell">No open invoices found.</td></tr>'
      : state.rawPage.map(inv => `
          <tr class="fade-in">
            <td>${inv.invoice_id ?? '—'}</td>
            <td>${inv.name_customer ?? '—'}</td>
            <td class="num">${fmt_usd(inv.total_open_amount)}</td>
            <td class="num">${fmt_date(inv.due_in_date)}</td>
            <td class="num"><span class="pred-date-pending">—</span></td>
            <td class="num"><span class="conf-range">—</span></td>
          </tr>`).join('');
}

/* ════════════════════════════════════════════════════════════
   POST-PREDICTION: invoice table (client-side pagination)
   ════════════════════════════════════════════════════════════ */
function renderPredictedInvoiceTable(page) {
  state.invPage = page;
  const slice = state.invoices.slice((page - 1) * state.PAGE_SIZE, page * state.PAGE_SIZE);

  document.getElementById('invoice-tbody').innerHTML = slice.map(inv => {
    const late      = parseInt(inv.days_late_pred ?? 0);
    const dateClass = late > 0 ? 'pred-date-late' : 'pred-date-ontime';
    const delayStr  = late > 0
      ? `<span class="delay-badge" style="color:var(--red)">(+${late}d)</span>`
      : `<span class="delay-badge" style="color:var(--green)">(on time)</span>`;

    const rangeStr = inv.predicted_payment_lower && inv.predicted_payment_upper
      ? `<span class="conf-range">${fmt_date(inv.predicted_payment_lower)} — ${fmt_date(inv.predicted_payment_upper)}</span>`
      : '<span class="conf-range">—</span>';

    return `
      <tr class="fade-in">
        <td>${inv.invoice_id ?? '—'}</td>
        <td>${inv.name_customer ?? '—'}</td>
        <td class="num">${fmt_usd(inv.total_open_amount)}</td>
        <td class="num">${fmt_date(inv.due_in_date)}</td>
        <td class="num">
          <span class="${dateClass}">${fmt_date(inv.predicted_payment_date)}</span>${delayStr}
        </td>
        <td class="num">${rangeStr}</td>
      </tr>`;
  }).join('');

  renderPagination('pagination', page,
    Math.ceil(state.invoices.length / state.PAGE_SIZE),
    state.invoices.length, 'renderPredictedInvoiceTable');
}

/* ════════════════════════════════════════════════════════════
   PRIORITY TABLE (client-side pagination)
   ════════════════════════════════════════════════════════════ */
function renderPriorityTable(page) {
  state.priPage = page;
  document.getElementById('priority-empty').style.display = 'none';
  document.getElementById('priority-table').style.display = 'table';

  const total = state.priority.length;
  const slice = state.priority.slice((page - 1) * state.PAGE_SIZE, page * state.PAGE_SIZE);

  document.getElementById('priority-badge').textContent = `${total} invoices ranked`;

  document.getElementById('priority-tbody').innerHTML = slice.map(row => {
    const ev      = parseFloat(row.expected_value ?? 0);
    const evClass = ev >= 0 ? 'ev-pos' : 'ev-neg';
    const evStr   = (ev >= 0 ? '+' : '') + fmt_usd(ev);
    const rankCls = { CALL: 'rank-call', REMIND: 'rank-remind', WATCH: 'rank-watch' }[row.action] || '';
    return `
      <tr class="fade-in">
        <td class="num"><span class="${rankCls}">${row.rank}</span></td>
        <td>${row.name_customer ?? '—'}</td>
        <td class="num">${fmt_usd(row.total_open_amount)}</td>
        <td class="num">${delayCell(row.days_late_pred)}</td>
        <td class="num">${probCell(row.p_late)}</td>
        <td class="num">${probCell(row.p_responds)}</td>
        <td class="num"><span class="${evClass}">${evStr}</span></td>
        <td class="center">${actionBadge(row.action)}</td>
      </tr>`;
  }).join('');

  renderPagination('priority-pagination', page,
    Math.ceil(total / state.PAGE_SIZE), total, 'renderPriorityTable');
}

/* ════════════════════════════════════════════════════════════
   KPI UPDATE
   ════════════════════════════════════════════════════════════ */
function updateKPIs(kpis) {
  document.getElementById('kpi-avg-late-val').textContent = `${kpis.avg_days_late}d`;
  document.getElementById('kpi-at-risk-val').textContent  = fmt_num(kpis.total_at_risk);
  document.getElementById('kpi-avg-late').style.opacity   = '1';
  document.getElementById('kpi-at-risk').style.opacity    = '1';

  const priKpis = document.getElementById('priority-kpis');
  priKpis.style.opacity = '1';
  priKpis.style.pointerEvents = 'auto';

  document.getElementById('kpi-call-count').textContent    = fmt_num(kpis.call_count);
  document.getElementById('kpi-remind-count').textContent  = fmt_num(kpis.remind_count);
  document.getElementById('kpi-watch-count').textContent   = fmt_num(kpis.watch_count);
  document.getElementById('kpi-daily-recovery').textContent = fmt_usd(kpis.daily_recovery);
}

/* ════════════════════════════════════════════════════════════
   PAGINATION RENDERER
   ════════════════════════════════════════════════════════════ */
function renderPagination(containerId, current, totalPages, totalItems, fnName) {
  const el = document.getElementById(containerId);
  if (!el || totalPages <= 1) { if (el) el.innerHTML = ''; return; }

  const MAX = 7;
  let pages = [];
  if (totalPages <= MAX) {
    pages = Array.from({ length: totalPages }, (_, i) => i + 1);
  } else {
    pages = [1];
    const s = Math.max(2, current - 2);
    const e = Math.min(totalPages - 1, current + 2);
    if (s > 2)              pages.push('…');
    for (let i = s; i <= e; i++) pages.push(i);
    if (e < totalPages - 1) pages.push('…');
    pages.push(totalPages);
  }

  const from = (current - 1) * state.PAGE_SIZE + 1;
  const to   = Math.min(current * state.PAGE_SIZE, totalItems);

  const btn = (label, page, disabled, active) =>
    `<button class="page-btn${active ? ' active' : ''}"
       ${disabled ? 'disabled' : `onclick="${fnName}(${page})"`}>${label}</button>`;

  el.innerHTML =
    btn('&#8249;', current - 1, current === 1, false) +
    pages.map(p =>
      p === '…'
        ? '<span class="page-info">…</span>'
        : btn(p, p, false, p === current)
    ).join('') +
    btn('&#8250;', current + 1, current === totalPages, false) +
    `<span class="page-info">${from}–${to} / ${totalItems.toLocaleString()}</span>`;
}

/* ════════════════════════════════════════════════════════════
   PREDICT BUTTON
   ════════════════════════════════════════════════════════════ */
async function runPredictions() {
  const btn = document.getElementById('predict-btn');
  const txt = document.getElementById('btn-text');
  const ico = document.getElementById('btn-icon');

  btn.disabled = true;
  btn.classList.add('loading');
  txt.textContent = 'Running…';
  ico.innerHTML   = '<span class="spinner"></span>';

  try {
    const res  = await fetch('/api/predict', { method: 'POST' });
    const data = await res.json();

    state.invoices  = data.invoices;
    state.priority  = data.priority;
    state.predicted = true;

    renderPredictedInvoiceTable(1);
    renderPriorityTable(1);
    updateKPIs(data.kpis);

    if (data.strategy_comparison) {
      await waitForChartJs();
      renderImpactFromPredict(data.strategy_comparison, data.kpis);
    }

    btn.classList.remove('loading');
    btn.classList.add('done');
    txt.textContent = 'Predictions Ready';
    ico.innerHTML   = `<svg viewBox="0 0 20 20" fill="currentColor">
      <path fill-rule="evenodd"
        d="M16.707 5.293a1 1 0 010 1.414L8.414 15l-4.707-4.707a1 1
           0 011.414-1.414L8.414 12.172l7.879-7.879a1 1 0 011.414 0z"
        clip-rule="evenodd"/>
    </svg>`;

    document.getElementById('last-run').textContent =
      `Last run: ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;

  } catch (err) {
    btn.disabled = false;
    btn.classList.remove('loading');
    txt.textContent = 'Run Predictions';
    ico.innerHTML   = `<svg viewBox="0 0 20 20" fill="currentColor">
      <path fill-rule="evenodd"
        d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1
           1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"/>
    </svg>`;
    console.error('Prediction failed:', err);
  }
}

/* ════════════════════════════════════════════════════════════
   ANALYTICS — load on startup, render both analytics tabs
   ════════════════════════════════════════════════════════════ */
async function waitForChartJs() {
  return new Promise(resolve => {
    const check = () => {
      if (typeof Chart !== 'undefined') { initChartDefaults(); resolve(); }
      else setTimeout(check, 50);
    };
    check();
  });
}

async function loadAnalytics() {
  try {
    const res  = await fetch('/api/analytics');
    const data = await res.json();
    state.analytics = data;
    await waitForChartJs();
    renderIntelligenceTab(data);
  } catch (err) {
    console.error('Analytics load failed:', err);
  }
}

/* ─── Chart.js dark defaults (set once) ─────────────────── */
function initChartDefaults() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.color        = '#64748b';
  Chart.defaults.borderColor  = '#1a3055';
  Chart.defaults.font.family  = "'Inter','Segoe UI',system-ui,sans-serif";
  Chart.defaults.font.size    = 11;
}

function destroyChart(id) {
  if (state.charts[id]) {
    state.charts[id].destroy();
    delete state.charts[id];
  }
}

/* ════════════════════════════════════════════════════════════
   IMPACT TAB — populated from /api/predict response
   ════════════════════════════════════════════════════════════ */
function renderImpactFromPredict(strategyData, kpis) {
  // Impact KPI cards
  const engine = strategyData.strategies.find(s => s.name === 'Decision Engine');
  const random = strategyData.strategies.find(s => s.name === 'Random Selection');
  const amount = strategyData.strategies.find(s => s.name === 'By Invoice Size');

  if (engine) {
    document.getElementById('imp-annual-engine').textContent = fmt_usd_short(engine.annual_ev);
    document.getElementById('impact-kpis').style.opacity = '1';
  }
  document.getElementById('imp-vs-random').textContent = fmt_usd_short(strategyData.improvement_vs_random);
  document.getElementById('imp-vs-amount').textContent = fmt_usd_short(strategyData.improvement_vs_amount);

  renderStrategyChart(strategyData.strategies);

  if (kpis && kpis.tier_exposure) {
    renderVarChart(kpis.tier_exposure);
  }
}

function renderStrategyChart(strategies) {
  const wrap = document.getElementById('strategy-chart-wrap');
  const msg  = document.getElementById('strategy-empty-msg');
  wrap.style.display = 'block';
  msg.style.display  = 'none';

  destroyChart('strategy');
  const ctx = document.getElementById('strategy-chart').getContext('2d');

  const labels = strategies.map(s => s.name);
  const values = strategies.map(s => s.daily_ev);
  const colors = strategies.map(s => s.color);

  state.charts['strategy'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors.map(c => c + 'cc'),
        borderColor:     colors,
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${fmt_usd(ctx.raw)}/day  →  ${fmt_usd_short(ctx.raw * 250)}/yr`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: '#1a3055' },
          ticks: {
            callback: v => fmt_usd_short(v),
            color: '#64748b',
          },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#94a3b8' },
        },
      },
    },
  });
}

function renderVarChart(tierExposure) {
  const wrap = document.getElementById('var-chart-wrap');
  const msg  = document.getElementById('var-empty-msg');
  const leg  = document.getElementById('var-legend');
  wrap.style.display = 'block';
  leg.style.display  = 'flex';
  msg.style.display  = 'none';

  destroyChart('var');
  const ctx = document.getElementById('var-chart').getContext('2d');

  const TIER_META = {
    call:   { label: 'CALL',   color: '#f87171' },
    remind: { label: 'REMIND', color: '#fbbf24' },
    watch:  { label: 'WATCH',  color: '#60a5fa' },
    ok:     { label: 'OK',     color: '#10b981' },
  };

  const labels = Object.keys(TIER_META).map(k => TIER_META[k].label);
  const values = Object.keys(TIER_META).map(k => tierExposure[k] ?? 0);
  const colors = Object.keys(TIER_META).map(k => TIER_META[k].color);

  state.charts['var'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors.map(c => c + 'bb'),
        borderColor: colors,
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '62%',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${fmt_usd_short(ctx.raw)} (${((ctx.raw / values.reduce((a,b)=>a+b,0))*100).toFixed(1)}%)`,
          },
        },
      },
    },
  });

  leg.innerHTML = Object.keys(TIER_META).map((k, i) => `
    <div class="var-legend-item">
      <span class="var-legend-dot" style="background:${colors[i]}"></span>
      ${labels[i]}: ${fmt_usd_short(values[i])}
    </div>`).join('');
}

/* ════════════════════════════════════════════════════════════
   MODEL INTELLIGENCE TAB
   ════════════════════════════════════════════════════════════ */
function renderIntelligenceTab(analytics) {
  renderModelMetrics(analytics.interval_stats);
  renderFeatureImportanceChart(analytics.feature_importance);
  renderCustomerScatter(analytics.customer_segments);
  renderSegmentCards(analytics.customer_segments);
}

function renderModelMetrics(stats) {
  document.getElementById('intel-rmse').textContent    = stats.rmse?.toFixed(2) ?? '—';
  document.getElementById('intel-mae').textContent     = stats.mae?.toFixed(2) ?? '—';
  document.getElementById('intel-within3').textContent = fmt_pct(stats.within_3_days_pct);
  document.getElementById('intel-pi-cov').textContent  = fmt_pct(stats.pi_coverage);
  document.getElementById('intel-test-size').textContent =
    `${(stats.total_test_invoices ?? 0).toLocaleString()} test invoices`;

  document.getElementById('pi-coverage-val').textContent = fmt_pct(stats.pi_coverage);
  document.getElementById('pi-cal-badge').textContent    = stats.calibration_label ?? '—';
  document.getElementById('pi-width-val').textContent    = stats.mean_pi_width?.toFixed(1) ?? '—';
  document.getElementById('pi-median-val').textContent   = stats.median_pi_width?.toFixed(1) ?? '—';

  // Coverage buckets table
  const tbody = document.getElementById('coverage-tbody');
  if (stats.coverage_by_width_bucket?.length) {
    tbody.innerHTML = stats.coverage_by_width_bucket.map(b => {
      const cov = b.coverage ?? 0;
      const barW = Math.round(cov);
      return `
        <tr>
          <td>${b.bucket}</td>
          <td style="text-align:right;color:var(--text-muted);">${b.count.toLocaleString()}</td>
          <td style="text-align:right;">${cov.toFixed(1)}%</td>
          <td style="width:80px;">
            <div class="cov-bar-wrap">
              <div class="cov-bar" style="width:${barW}%;max-width:70px;"></div>
            </div>
          </td>
        </tr>`;
    }).join('');
  }

  // Model Architecture
  const p = stats.model_params ?? {};
  const archEl = document.getElementById('model-arch-rows');
  const rows = [
    ['Algorithm',       'Gradient Boosting Regressor (scikit-learn)'],
    ['Estimators',      p.n_estimators ?? 100],
    ['Max Depth',       p.max_depth ?? 5],
    ['Learning Rate',   p.learning_rate ?? 0.1],
    ['Models',          'Mean + Quantile α=0.1 + Quantile α=0.9'],
    ['Features',        '13 (6 customer history, 4 temporal, 2 categorical, 1 amount)'],
    ['Test Set',        `${(stats.total_test_invoices ?? 0).toLocaleString()} invoices (20% holdout)`],
  ];
  archEl.innerHTML = rows.map(([k, v]) =>
    `<div class="model-arch-row">
       <span class="model-arch-label">${k}</span>
       <span class="model-arch-value">${v}</span>
     </div>`).join('');
}

function renderFeatureImportanceChart(data) {
  destroyChart('features');
  const ctx = document.getElementById('feature-importance-chart').getContext('2d');

  const features = data.features;
  const labels   = features.map(f => f.label);
  const values   = features.map(f => +(f.mean_importance * 100).toFixed(1));
  const colors   = features.map(f => f.color);

  state.charts['features'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors.map(c => c + 'bb'),
        borderColor:     colors,
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.raw.toFixed(1)}% importance`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: '#1a3055' },
          ticks: { callback: v => v + '%', color: '#64748b' },
          max: Math.ceil(Math.max(...values) * 1.1),
        },
        y: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 10 } },
        },
      },
    },
  });

  // Category pills
  const pills = document.getElementById('category-pills');
  const catColors = {
    customer_history: '#3b82f6',
    amount:           '#10b981',
    categorical:      '#fbbf24',
    temporal:         '#60a5fa',
  };
  const catLabels = {
    customer_history: 'Customer History',
    amount:           'Amount',
    categorical:      'Terms & Business Unit',
    temporal:         'Temporal',
  };
  pills.innerHTML = Object.entries(data.categories)
    .sort((a, b) => b[1] - a[1])
    .map(([cat, imp]) => {
      const col = catColors[cat] || '#888';
      return `<span class="category-pill" style="color:${col};border-color:${col}44;background:${col}11;">
        ${catLabels[cat] || cat} ${(imp * 100).toFixed(1)}%
      </span>`;
    }).join('');
}

function renderCustomerScatter(data) {
  destroyChart('scatter');
  const ctx = document.getElementById('customer-scatter-chart').getContext('2d');

  const segColors = {
    reliable:         '#10b981',
    consistently_late:'#f87171',
    volatile:         '#fbbf24',
    high_risk:        '#fb923c',
  };

  // Group scatter points by segment
  const grouped = {};
  data.scatter_data.forEach(pt => {
    if (!grouped[pt.segment]) grouped[pt.segment] = [];
    grouped[pt.segment].push({ x: pt.x, y: pt.y });
  });

  const datasets = data.segments.map(seg => ({
    label:           seg.label,
    data:            grouped[seg.id] || [],
    backgroundColor: seg.color + '88',
    borderColor:     seg.color,
    borderWidth:     0.5,
    pointRadius:     3,
    pointHoverRadius:5,
  }));

  const threshX = data.thresholds.avg_days_late_median;
  const threshY = data.thresholds.std_days_late_median;

  // Custom plugin to draw threshold lines
  const thresholdPlugin = {
    id: 'thresholds',
    afterDraw(chart) {
      const { ctx: c, chartArea, scales } = chart;
      const xPx = scales.x.getPixelForValue(threshX);
      const yPx = scales.y.getPixelForValue(threshY);
      c.save();
      c.strokeStyle = '#1e3a60';
      c.lineWidth   = 1;
      c.setLineDash([4, 4]);
      // vertical line at x=median
      c.beginPath();
      c.moveTo(xPx, chartArea.top);
      c.lineTo(xPx, chartArea.bottom);
      c.stroke();
      // horizontal line at y=median
      c.beginPath();
      c.moveTo(chartArea.left, yPx);
      c.lineTo(chartArea.right, yPx);
      c.stroke();
      c.restore();
    },
  };

  state.charts['scatter'] = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
          labels: { boxWidth: 10, padding: 10, font: { size: 10 } },
        },
        tooltip: {
          callbacks: {
            label: ctx => `Avg: ${ctx.raw.x}d, Std: ${ctx.raw.y}d`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Avg Days Late', color: '#64748b', font: { size: 10 } },
          grid:  { color: '#1a3055' },
          ticks: { color: '#64748b' },
          min:   -25, max: 40,
        },
        y: {
          title: { display: true, text: 'Payment Variability (std)', color: '#64748b', font: { size: 10 } },
          grid:  { color: '#1a3055' },
          ticks: { color: '#64748b' },
          min:    0,  max: 45,
        },
      },
    },
    plugins: [thresholdPlugin],
  });

  document.getElementById('seg-customer-count').textContent =
    data.scatter_data.length.toLocaleString();
}

function renderSegmentCards(data) {
  const grid = document.getElementById('segment-cards');
  grid.innerHTML = data.segments.map(seg => `
    <div class="segment-card" style="border-left-color:${seg.color}">
      <div class="segment-card-label" style="color:${seg.color}">${seg.label}</div>
      <div class="segment-card-count">${seg.count}</div>
      <div class="segment-card-stat">Avg delay: ${seg.avg_days_late > 0 ? '+' : ''}${seg.avg_days_late}d</div>
      <div class="segment-card-stat">Std: ${seg.avg_std}d</div>
    </div>`).join('');
}

/* ─── Init ───────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  requestAnimationFrame(() => {
    state.PAGE_SIZE = calcPageSize();
    fetchRawPage(1);
    loadAnalytics();
  });
});
