"""
Module 6 — Interactive Dashboard Generator
===========================================
Generates a single-file HTML dashboard with Plotly charts covering:
  - Spread reconstruction comparison (Roll vs C-S vs L2 vs True)
  - Microstructure feature time series
  - Model predictions vs actual
  - Backtest P&L curve with inventory and regime overlays
  - P&L attribution (spread income vs inventory PnL)

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any


def df_to_json(df: pd.DataFrame, cols=None) -> str:
    """Serialize DataFrame columns to JSON-compatible dict."""
    if cols:
        df = df[cols]
    result = {}
    for col in df.columns:
        vals = df[col]
        if hasattr(vals, 'dt'):
            result[col] = vals.dt.strftime('%H:%M:%S').tolist()
        elif vals.dtype == object or str(vals.dtype) in ('string', 'str') or vals.apply(lambda x: isinstance(x, str)).any():
            result[col] = vals.tolist()
        else:
            result[col] = [round(float(v), 6) if pd.notna(v) else None for v in vals]
    return json.dumps(result)


def generate_dashboard(
    df_ticks: pd.DataFrame,
    df_reconstruction: pd.DataFrame,
    df_features: pd.DataFrame,
    df_backtest: pd.DataFrame,
    cv_results: pd.DataFrame,
    output_path: str = 'dashboard.html',
) -> str:
    """
    Generate full pipeline dashboard as a standalone HTML file.
    """
    # Subsample for performance
    N = min(2000, len(df_backtest))
    idx = np.linspace(0, len(df_backtest)-1, N, dtype=int)
    bt = df_backtest.iloc[idx].copy()
    bt['timestamp_str'] = bt['timestamp'].dt.strftime('%H:%M:%S')

    N2 = min(1000, len(df_reconstruction))
    idx2 = np.linspace(0, len(df_reconstruction)-1, N2, dtype=int)
    rec = df_reconstruction.iloc[idx2].copy()
    if hasattr(rec.index, 'strftime'):
        rec['ts'] = rec.index.strftime('%H:%M:%S')
    else:
        rec['ts'] = rec.index.astype(str)

    N3 = min(1000, len(df_features))
    idx3 = np.linspace(0, len(df_features)-1, N3, dtype=int)
    feat = df_features.iloc[idx3].copy()
    feat['ts'] = feat['timestamp'].dt.strftime('%H:%M:%S')

    # Serialize data
    bt_data  = df_to_json(bt,   ['timestamp_str','mid','bid','ask','quoted_spread',
                                   'predicted_spread','true_spread','mtm_pnl',
                                   'spread_income','inventory_pnl','inventory','vpin','vol_regime'])
    rec_data = df_to_json(rec,  ['ts','true_spread','roll_spread','cs_spread','l2_spread'])
    feat_data= df_to_json(feat, ['ts','vpin','ofi','kyle_lambda','realized_vol'])
    cv_data  = df_to_json(cv_results, ['fold','model','mae','rmse','r2'])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bid-Ask Replication Pipeline — Market Microstructure Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #08090d;
    --bg2: #0f1117;
    --bg3: #171a23;
    --border: #1e2235;
    --accent: #00d4ff;
    --accent2: #7b61ff;
    --accent3: #00ff99;
    --accent4: #ff6b6b;
    --accent5: #ffaa00;
    --text: #e8eaf0;
    --text2: #8892a4;
    --text3: #4a5568;
    --green: #00e676;
    --red: #ff5252;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    line-height: 1.5;
    overflow-x: hidden;
  }}

  /* Scan-line overlay */
  body::before {{
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.03) 2px,
      rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 1000;
  }}

  header {{
    padding: 28px 32px 20px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, var(--bg2), var(--bg));
    position: relative;
    overflow: hidden;
  }}

  header::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
  }}

  .header-grid {{
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: start;
    gap: 16px;
    max-width: 1600px;
  }}

  .header-title {{
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}

  h1 {{
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1.2;
  }}

  h1 span {{ color: var(--accent); }}

  .header-sub {{
    font-size: 12px;
    color: var(--text2);
    margin-top: 6px;
    font-family: var(--mono);
  }}

  .badges {{
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
  }}

  .badge {{
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 2px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}

  .badge-blue   {{ background: rgba(0,212,255,0.1); color: var(--accent); border: 1px solid rgba(0,212,255,0.2); }}
  .badge-purple {{ background: rgba(123,97,255,0.1); color: var(--accent2); border: 1px solid rgba(123,97,255,0.2); }}
  .badge-green  {{ background: rgba(0,255,153,0.1); color: var(--accent3); border: 1px solid rgba(0,255,153,0.2); }}
  .badge-red    {{ background: rgba(255,107,107,0.1); color: var(--accent4); border: 1px solid rgba(255,107,107,0.2); }}

  .stats-strip {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    border-bottom: 1px solid var(--border);
    background: var(--bg2);
  }}

  .stat {{
    padding: 14px 20px;
    border-right: 1px solid var(--border);
    position: relative;
  }}

  .stat:last-child {{ border-right: none; }}

  .stat-label {{
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text3);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}

  .stat-value {{
    font-family: var(--mono);
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
  }}

  .stat-value.pos {{ color: var(--green); }}
  .stat-value.neg {{ color: var(--red); }}
  .stat-value.neutral {{ color: var(--accent); }}

  .stat-sub {{
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text3);
    margin-top: 2px;
  }}

  .main {{
    max-width: 1600px;
    padding: 24px 32px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}

  .card {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
    position: relative;
  }}

  .card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.3;
  }}

  .card.full-width {{
    grid-column: 1 / -1;
  }}

  .card-header {{
    padding: 12px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg3);
  }}

  .card-title {{
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    color: var(--text2);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}

  .card-tag {{
    font-family: var(--mono);
    font-size: 8px;
    padding: 2px 7px;
    background: rgba(0,212,255,0.08);
    color: var(--accent);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 1px;
    letter-spacing: 0.08em;
  }}

  .chart-container {{
    padding: 8px;
    height: 300px;
  }}

  .chart-container.tall {{
    height: 360px;
  }}

  .chart-container.short {{
    height: 240px;
  }}

  .section-label {{
    grid-column: 1 / -1;
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text3);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 8px 0 4px;
    border-top: 1px solid var(--border);
    margin-top: 4px;
  }}

  footer {{
    border-top: 1px solid var(--border);
    padding: 16px 32px;
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text3);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg2);
  }}

  .footer-links {{
    display: flex;
    gap: 16px;
  }}

  .footer-links span {{
    color: var(--text3);
  }}
</style>
</head>
<body>

<header>
  <div class="header-grid">
    <div>
      <div class="header-title">// Market Microstructure Research</div>
      <h1>Bid-Ask <span>Replication</span> Pipeline</h1>
      <div class="header-sub">Tick Classification → Spread Reconstruction → Feature Engineering → Modelling → Optimal Quoting</div>
      <div class="badges">
        <span class="badge badge-blue">Lee-Ready (1991)</span>
        <span class="badge badge-blue">Roll (1984)</span>
        <span class="badge badge-blue">Corwin-Schultz (2012)</span>
        <span class="badge badge-purple">VPIN · OFI · Kyle λ</span>
        <span class="badge badge-green">OLS · Ridge · GBM</span>
        <span class="badge badge-red">Avellaneda-Stoikov (2008)</span>
      </div>
    </div>
    <div style="text-align:right; font-family: var(--mono); font-size:9px; color: var(--text3); line-height:2">
      <div>Philippe-Emmanuel Yao</div>
      <div>MSc Financial Mathematics · LSE</div>
      <div style="color: var(--accent); margin-top:4px">github.com/philippeyao</div>
    </div>
  </div>
</header>

<div class="stats-strip" id="stats-strip">
  <div class="stat">
    <div class="stat-label">Total P&amp;L</div>
    <div class="stat-value pos" id="stat-pnl">—</div>
    <div class="stat-sub">Mark-to-market</div>
  </div>
  <div class="stat">
    <div class="stat-label">Sharpe Ratio</div>
    <div class="stat-value neutral" id="stat-sharpe">—</div>
    <div class="stat-sub">Annualized</div>
  </div>
  <div class="stat">
    <div class="stat-label">Max Drawdown</div>
    <div class="stat-value neg" id="stat-dd">—</div>
    <div class="stat-sub">Peak-to-trough</div>
  </div>
  <div class="stat">
    <div class="stat-label">Total Fills</div>
    <div class="stat-value neutral" id="stat-fills">—</div>
    <div class="stat-sub">Round-trips</div>
  </div>
  <div class="stat">
    <div class="stat-label">Avg |Inventory|</div>
    <div class="stat-value" id="stat-inv">—</div>
    <div class="stat-sub">Units held</div>
  </div>
  <div class="stat">
    <div class="stat-label">Spread Income</div>
    <div class="stat-value pos" id="stat-spread-inc">—</div>
    <div class="stat-sub">Gross capture</div>
  </div>
</div>

<div class="main">

  <div class="section-label">// MODULE 2 — SPREAD RECONSTRUCTION</div>

  <div class="card full-width">
    <div class="card-header">
      <span class="card-title">Bid-Ask Spread Estimators vs True Spread</span>
      <span class="card-tag">Roll · Corwin-Schultz · Level-2 Proxy</span>
    </div>
    <div class="chart-container tall" id="chart-reconstruction"></div>
  </div>

  <div class="section-label">// MODULE 3 — MICROSTRUCTURE FEATURES</div>

  <div class="card">
    <div class="card-header">
      <span class="card-title">VPIN — Informed Trading Probability</span>
      <span class="card-tag">Easley et al. 2012</span>
    </div>
    <div class="chart-container short" id="chart-vpin"></div>
  </div>

  <div class="card">
    <div class="card-header">
      <span class="card-title">Order Flow Imbalance &amp; Kyle's Lambda</span>
      <span class="card-tag">Cont et al. 2014 · Kyle 1985</span>
    </div>
    <div class="chart-container short" id="chart-ofi-lambda"></div>
  </div>

  <div class="section-label">// MODULE 4 — SPREAD MODELLING</div>

  <div class="card">
    <div class="card-header">
      <span class="card-title">Walk-Forward CV — Model Comparison</span>
      <span class="card-tag">OLS · Ridge · GBM</span>
    </div>
    <div class="chart-container" id="chart-cv"></div>
  </div>

  <div class="card">
    <div class="card-header">
      <span class="card-title">Spread Prediction vs Actual (Test Set)</span>
      <span class="card-tag">Out-of-Sample</span>
    </div>
    <div class="chart-container" id="chart-predictions"></div>
  </div>

  <div class="section-label">// MODULE 5 — OPTIMAL QUOTING BACKTEST</div>

  <div class="card full-width">
    <div class="card-header">
      <span class="card-title">Backtest P&amp;L — MTM with Spread Income &amp; Inventory Components</span>
      <span class="card-tag">Avellaneda-Stoikov</span>
    </div>
    <div class="chart-container tall" id="chart-pnl"></div>
  </div>

  <div class="card">
    <div class="card-header">
      <span class="card-title">Inventory &amp; VPIN Over Time</span>
      <span class="card-tag">Risk Monitor</span>
    </div>
    <div class="chart-container" id="chart-inventory"></div>
  </div>

  <div class="card">
    <div class="card-header">
      <span class="card-title">Quoted vs True vs Predicted Spread</span>
      <span class="card-tag">Quote Quality</span>
    </div>
    <div class="chart-container" id="chart-spreads"></div>
  </div>

</div>

<footer>
  <span>Bid-Ask Replication Pipeline · Philippe-Emmanuel Yao · MSc Financial Mathematics, LSE · 2026</span>
  <div class="footer-links">
    <span>Lee-Ready Classification</span>
    <span>Roll / Corwin-Schultz / L2</span>
    <span>VPIN · OFI · Kyle λ</span>
    <span>OLS · Ridge · GBM</span>
    <span>Avellaneda-Stoikov Optimal Quoting</span>
  </div>
</footer>

<script>
const BT   = {bt_data};
const REC  = {rec_data};
const FEAT = {feat_data};
const CV   = {cv_data};

const PALETTE = {{
  accent:  '#00d4ff',
  accent2: '#7b61ff',
  accent3: '#00ff99',
  accent4: '#ff6b6b',
  accent5: '#ffaa00',
  grid:    '#1e2235',
  bg:      '#0f1117',
  bg3:     '#171a23',
  text:    '#e8eaf0',
  text2:   '#8892a4',
}};

const baseLayout = {{
  paper_bgcolor: 'transparent',
  plot_bgcolor:  PALETTE.bg,
  font: {{ family: 'IBM Plex Mono', color: PALETTE.text2, size: 10 }},
  margin: {{ t: 8, r: 16, b: 36, l: 52 }},
  xaxis: {{
    gridcolor: PALETTE.grid, gridwidth: 0.5,
    linecolor: PALETTE.grid, tickcolor: PALETTE.grid,
    tickfont: {{ size: 9 }},
    showgrid: true,
  }},
  yaxis: {{
    gridcolor: PALETTE.grid, gridwidth: 0.5,
    linecolor: PALETTE.grid, tickcolor: PALETTE.grid,
    tickfont: {{ size: 9 }},
    showgrid: true, zeroline: false,
  }},
  legend: {{
    bgcolor: 'rgba(15,17,23,0.8)',
    bordercolor: PALETTE.grid,
    borderwidth: 1,
    font: {{ size: 9, color: PALETTE.text2 }},
    orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1,
  }},
  hovermode: 'x unified',
  hoverlabel: {{
    bgcolor: '#171a23',
    bordercolor: PALETTE.accent,
    font: {{ family: 'IBM Plex Mono', size: 10 }},
  }},
}};

const cfg = {{ responsive: true, displayModeBar: false }};

// --- Stats Strip ---
const pnl = BT.mtm_pnl;
const finalPnl = pnl[pnl.length - 1];
const returns = pnl.slice(1).map((v, i) => v - pnl[i]);
const mean_r = returns.reduce((a,b)=>a+b,0)/returns.length;
const std_r  = Math.sqrt(returns.map(r=>(r-mean_r)**2).reduce((a,b)=>a+b,0)/returns.length);
const sharpe = (mean_r / (std_r + 1e-10)) * Math.sqrt(252 * 3600);
let maxPnl = -Infinity, maxDD = 0;
pnl.forEach(v => {{
  if (v > maxPnl) maxPnl = v;
  const dd = v - maxPnl;
  if (dd < maxDD) maxDD = dd;
}});
const totalFills = BT.n_fills ? BT.n_fills.reduce((a,b)=>a+b,0) : 0;
const avgInv = BT.inventory ? BT.inventory.map(Math.abs).reduce((a,b)=>a+b,0)/BT.inventory.length : 0;
const spreadInc = BT.spread_income ? BT.spread_income.reduce((a,b)=>a+b,0) : 0;

document.getElementById('stat-pnl').textContent = (finalPnl >= 0 ? '+' : '') + finalPnl.toFixed(1);
document.getElementById('stat-sharpe').textContent = sharpe.toFixed(2);
document.getElementById('stat-dd').textContent = maxDD.toFixed(1);
document.getElementById('stat-fills').textContent = totalFills.toLocaleString();
document.getElementById('stat-inv').textContent = avgInv.toFixed(1);
document.getElementById('stat-spread-inc').textContent = (spreadInc >= 0 ? '+' : '') + spreadInc.toFixed(2);

// --- Chart 1: Spread Reconstruction ---
Plotly.newPlot('chart-reconstruction', [
  {{
    x: REC.ts, y: REC.true_spread,
    name: 'True Spread', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent, width: 1.5 }},
  }},
  {{
    x: REC.ts, y: REC.roll_spread,
    name: 'Roll (1984)', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent4, width: 1, dash: 'dash' }},
  }},
  {{
    x: REC.ts, y: REC.cs_spread,
    name: 'Corwin-Schultz', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent2, width: 1, dash: 'dot' }},
  }},
  {{
    x: REC.ts, y: REC.l2_spread,
    name: 'Level-2 Proxy', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent3, width: 1.5 }},
  }},
], {{
  ...baseLayout,
  yaxis: {{ ...baseLayout.yaxis, title: {{ text: 'Spread', font: {{ size: 9 }} }} }},
}}, cfg);

// --- Chart 2: VPIN ---
Plotly.newPlot('chart-vpin', [
  {{
    x: FEAT.ts, y: FEAT.vpin,
    name: 'VPIN', type: 'scatter', mode: 'lines',
    fill: 'tozeroy',
    fillcolor: 'rgba(255,107,107,0.08)',
    line: {{ color: PALETTE.accent4, width: 1.5 }},
  }},
  {{
    x: FEAT.ts, y: FEAT.ts.map(() => 0.5),
    name: 'Threshold (0.5)', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent5, width: 1, dash: 'dash' }},
  }},
], {{
  ...baseLayout,
  yaxis: {{ ...baseLayout.yaxis, range: [0, 1], title: {{ text: 'VPIN', font: {{ size: 9 }} }} }},
}}, cfg);

// --- Chart 3: OFI + Kyle Lambda ---
Plotly.newPlot('chart-ofi-lambda', [
  {{
    x: FEAT.ts, y: FEAT.ofi,
    name: 'OFI', type: 'bar',
    marker: {{ color: FEAT.ofi.map(v => v >= 0 ? 'rgba(0,230,118,0.5)' : 'rgba(255,82,82,0.5)') }},
    yaxis: 'y',
  }},
  {{
    x: FEAT.ts, y: FEAT.kyle_lambda,
    name: "Kyle's λ", type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent2, width: 1.5 }},
    yaxis: 'y2',
  }},
], {{
  ...baseLayout,
  yaxis:  {{ ...baseLayout.yaxis, title: {{ text: 'OFI', font: {{ size: 9 }} }} }},
  yaxis2: {{ overlaying: 'y', side: 'right', title: {{ text: "Kyle λ", font: {{ size: 9 }} }},
    gridcolor: 'transparent', tickfont: {{ size: 9 }}, showgrid: false }},
}}, cfg);

// --- Chart 4: Walk-forward CV ---
const models = [...new Set(CV.model)];
const colors = [PALETTE.accent, PALETTE.accent2, PALETTE.accent3];
const metricTraces = models.map((m, i) => {{
  const idx = CV.model.map((v,j) => v === m ? j : -1).filter(j => j >= 0);
  return {{
    name: m,
    x: idx.map(j => `Fold ${{CV.fold[j]}}`),
    y: idx.map(j => CV.r2[j]),
    type: 'bar',
    marker: {{ color: colors[i], opacity: 0.8 }},
  }};
}});

Plotly.newPlot('chart-cv', metricTraces, {{
  ...baseLayout,
  barmode: 'group',
  yaxis: {{ ...baseLayout.yaxis, title: {{ text: 'R² (OOS)', font: {{ size: 9 }} }}, range: [0, 1] }},
}}, cfg);

// --- Chart 5: Predictions vs Actual ---
const halfN = Math.floor(BT.timestamp_str.length / 2);
Plotly.newPlot('chart-predictions', [
  {{
    x: BT.timestamp_str.slice(halfN),
    y: BT.true_spread.slice(halfN),
    name: 'True Spread', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent, width: 1.5 }},
  }},
  {{
    x: BT.timestamp_str.slice(halfN),
    y: BT.predicted_spread.slice(halfN),
    name: 'Predicted (GBM)', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent3, width: 1.5, dash: 'dot' }},
  }},
], {{
  ...baseLayout,
  yaxis: {{ ...baseLayout.yaxis, title: {{ text: 'Spread', font: {{ size: 9 }} }} }},
}}, cfg);

// --- Chart 6: P&L ---
const cumSpread = BT.spread_income ? BT.spread_income.reduce((acc, v, i) => {{
  acc.push((acc[i-1] || 0) + v); return acc;
}}, []) : BT.mtm_pnl.map(() => 0);
const cumInv = BT.inventory_pnl ? BT.inventory_pnl.reduce((acc, v, i) => {{
  acc.push((acc[i-1] || 0) + v); return acc;
}}, []) : BT.mtm_pnl.map(() => 0);

Plotly.newPlot('chart-pnl', [
  {{
    x: BT.timestamp_str, y: BT.mtm_pnl,
    name: 'MTM P&L', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent, width: 2 }},
    fill: 'tozeroy', fillcolor: 'rgba(0,212,255,0.04)',
  }},
  {{
    x: BT.timestamp_str, y: cumSpread,
    name: 'Spread Income (cum.)', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent3, width: 1.5, dash: 'dash' }},
  }},
  {{
    x: BT.timestamp_str, y: cumInv,
    name: 'Inventory P&L (cum.)', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent4, width: 1, dash: 'dot' }},
  }},
], {{
  ...baseLayout,
  yaxis: {{ ...baseLayout.yaxis, title: {{ text: 'P&L', font: {{ size: 9 }} }}, zeroline: true, zerolinecolor: PALETTE.grid }},
}}, cfg);

// --- Chart 7: Inventory & VPIN ---
Plotly.newPlot('chart-inventory', [
  {{
    x: BT.timestamp_str, y: BT.inventory,
    name: 'Inventory', type: 'scatter', mode: 'lines',
    fill: 'tozeroy', fillcolor: 'rgba(123,97,255,0.08)',
    line: {{ color: PALETTE.accent2, width: 1.5 }},
  }},
  {{
    x: BT.timestamp_str, y: BT.vpin.map(v => v * 50),
    name: 'VPIN × 50', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent4, width: 1, dash: 'dot' }},
    yaxis: 'y2',
  }},
], {{
  ...baseLayout,
  yaxis:  {{ ...baseLayout.yaxis, title: {{ text: 'Inventory', font: {{ size: 9 }} }}, zeroline: true, zerolinecolor: PALETTE.grid }},
  yaxis2: {{ overlaying: 'y', side: 'right', title: {{ text: 'VPIN', font: {{ size: 9 }} }},
    gridcolor: 'transparent', tickfont: {{ size: 9 }}, range: [0, 1], showgrid: false }},
}}, cfg);

// --- Chart 8: Spread comparison ---
Plotly.newPlot('chart-spreads', [
  {{
    x: BT.timestamp_str, y: BT.true_spread,
    name: 'True Spread', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent, width: 1.5 }},
  }},
  {{
    x: BT.timestamp_str, y: BT.predicted_spread,
    name: 'Predicted', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent3, width: 1, dash: 'dot' }},
  }},
  {{
    x: BT.timestamp_str, y: BT.quoted_spread,
    name: 'Quoted (A-S)', type: 'scatter', mode: 'lines',
    line: {{ color: PALETTE.accent2, width: 1.5, dash: 'dash' }},
  }},
], {{
  ...baseLayout,
  yaxis: {{ ...baseLayout.yaxis, title: {{ text: 'Spread', font: {{ size: 9 }} }} }},
}}, cfg);

</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path
