#!/usr/bin/env python3
"""
Generate the Simplified M2 Capstone Architecture Flowchart (Type [A])
Faithfully styled after flowchart.png:
- Clean 3-row serpentine / snake flow (9 compact nodes)
- Minimal details: Tag badge, bold title, 1-line concise sublabel
- Standard pipeline (slate/white) vs Novel Causal Engine (coral/warm tint)
- Pure orthogonal routing, zero overlap, opaque label masks
- Deliverables:
  1. Flowcharts/m2_causal_architecture_simple.svg
  2. Flowcharts/m2_causal_architecture_simple.html (interactive micro-app: Pan/Zoom, Dark Mode, Export PNG/SVG, Tooltips)
  3. Flowcharts/m2_causal_architecture_simple.png (rendered via Google Chrome headless)
"""

import os
import subprocess
from pathlib import Path

# Paths
PRIMARY_DIR = Path("/home/oem/Documents/causal _graph_volatility")
ALT_DIR = Path("/home/oem/Documents/causal graph volatility")

# Canvas & Layout Geometry
WIDTH = 940
HEIGHT = 710
CONTAINER_X = 16
CONTAINER_Y = 16
CONTAINER_W = 908
CONTAINER_H = 678

# Card Geometry
CARD_W = 216
CARD_H = 76
CARD_R = 8

# Column X-coordinates (3 columns)
# Left margin = 48, Right margin = 48 -> Available = 940 - 96 = 844
# 3 * 216 = 648 -> Remaining = 196 -> Gap = 98
COL1_X = 48
COL2_X = 362
COL3_X = 676

# Row Y-coordinates (3 rows)
# Row 1: Y = 164..240 (center 202)
# Gap = 86 -> Row 2: Y = 326..402 (center 364)
# Gap = 86 -> Row 3: Y = 488..564 (center 526)
ROW1_Y = 164
ROW2_Y = 326
ROW3_Y = 488

# Node Definitions
# Type: 'standard' or 'novel'
NODES = [
    # Row 1 (Left to Right)
    {
        "id": "node1", "col": 1, "row": 1, "x": COL1_X, "y": ROW1_Y,
        "type": "standard", "tag": "INGEST", "tag_w": 52,
        "title": "Data Ingestion",
        "sublabel": "Equities & F-F Factors",
        "tooltip_title": "Stage 01: Multi-Asset & Systematic Ingestion",
        "tooltip_desc": "Continuous panel of S&P 100 constituent OHLCV data alongside Kenneth French 3-Factor (Mkt-RF, SMB, HML) and macro interest rate spreads."
    },
    {
        "id": "node2", "col": 2, "row": 1, "x": COL2_X, "y": ROW1_Y,
        "type": "standard", "tag": "RESIDUALIZE", "tag_w": 78,
        "title": "Factor Residualizer",
        "sublabel": "Vectorized OLS Engine",
        "tooltip_title": "Stage 02: Factor Purging Engine (Feedback A)",
        "tooltip_desc": "Regresses raw returns against Fama-French systematic factors: R_{i,t} = alpha_i + beta_i' F_t + epsilon_{i,t}. Isolates idiosyncratic residuals with Cov(epsilon, F) = 0."
    },
    {
        "id": "node3", "col": 3, "row": 1, "x": COL3_X, "y": ROW1_Y,
        "type": "standard", "tag": "BENCHMARK", "tag_w": 70,
        "title": "Precision Matrix",
        "sublabel": "Graphical LASSO (Θ)",
        "tooltip_title": "Stage 03: Undirected Independence Benchmark",
        "tooltip_desc": "L1-penalized sparse precision matrix Theta = Sigma^{-1}. Serves as the classic undirected conditional independence baseline for non-redundancy audits."
    },

    # Row 2 (Right to Left)
    {
        "id": "node4", "col": 3, "row": 2, "x": COL3_X, "y": ROW2_Y,
        "type": "novel", "tag": "CAUSAL DAG", "tag_w": 72,
        "title": "Causal Discovery",
        "sublabel": "Student-t DAGMA",
        "tooltip_title": "Stage 04: Continuous Non-Linear Causal Discovery",
        "tooltip_desc": "Learns directed acyclic graph W over idiosyncratic innovations under Student-t heavy-tailed loss with algebraic acyclicity constraint h(W) = 0."
    },
    {
        "id": "node5", "col": 2, "row": 2, "x": COL2_X, "y": ROW2_Y,
        "type": "novel", "tag": "VALIDATE", "tag_w": 62,
        "title": "Orthogonality Audit",
        "sublabel": "VIF & Non-Redundancy",
        "tooltip_title": "Stage 05: Empirical Non-Redundancy Audit (Feedback B)",
        "tooltip_desc": "Benchmarking DAG centrality (k_out) against Precision centrality (k_Theta). Verifies VIF < 5.0 and Spearman rank divergence to prove topological complementarity."
    },
    {
        "id": "node6", "col": 1, "row": 2, "x": COL1_X, "y": ROW2_Y,
        "type": "novel", "tag": "REGIME DRIFT", "tag_w": 82,
        "title": "Causal Drift Monitor",
        "sublabel": "Frobenius Velocity ΔW",
        "tooltip_title": "Stage 06: Structural Break & Regime Shift Monitor",
        "tooltip_desc": "Tracks rolling Frobenius causal velocity Delta W_t = ||W_t - W_{t-1}||_F. Flags structural dislocations (e.g. SVB collapse) when velocity breaches tau_crit."
    },

    # Row 3 (Left to Right)
    {
        "id": "node7", "col": 1, "row": 3, "x": COL1_X, "y": ROW3_Y,
        "type": "novel", "tag": "CATBOOST ML", "tag_w": 78,
        "title": "CatBoost Vol Regressor",
        "sublabel": "Student-t Heavy Tails",
        "tooltip_title": "Stage 07: Gradient Boosted Volatility Forecasting",
        "tooltip_desc": "GPU-accelerated CatBoost regression fusing classical Garman-Klass volatility, causal transmitter centrality, and structural break indicators under heavy-tailed loss."
    },
    {
        "id": "node8", "col": 2, "row": 3, "x": COL2_X, "y": ROW3_Y,
        "type": "novel", "tag": "PRUNING", "tag_w": 58,
        "title": "Contagion Pruner",
        "sublabel": "Hub Exits & Ratchet",
        "tooltip_title": "Stage 08: Asymmetric Contagion Pruning & Ratchet Stop",
        "tooltip_desc": "Pre-emptively prunes high out-degree transmitter hubs during stress, dynamically ratchets trailing stop buffer lambda_t in [2.0, 3.5], and re-enters on MA20 trend confirmation."
    },
    {
        "id": "node9", "col": 3, "row": 3, "x": COL3_X, "y": ROW3_Y,
        "type": "standard", "tag": "EVALUATE", "tag_w": 64,
        "title": "Strategy Evaluator",
        "sublabel": "Sharpe, Drawdowns & Alpha",
        "tooltip_title": "Stage 09: Out-of-Sample Performance Benchmarking",
        "tooltip_desc": "Evaluates walk-forward risk-adjusted returns, maximum drawdown containment, and execution frictions against Buy & Hold and Risk Parity benchmarks."
    }
]

# Connectors Definitions (8 clean straight segments)
CONNECTORS = [
    # Row 1: 1 -> 2 (Right)
    {
        "x1": COL1_X + CARD_W, "y1": ROW1_Y + CARD_H / 2,
        "x2": COL2_X, "y2": ROW1_Y + CARD_H / 2,
        "type": "standard", "label": "MARKET DATA", "label_w": 80
    },
    # Row 1: 2 -> 3 (Right)
    {
        "x1": COL2_X + CARD_W, "y1": ROW1_Y + CARD_H / 2,
        "x2": COL3_X, "y2": ROW1_Y + CARD_H / 2,
        "type": "standard", "label": "RESIDUALS ε", "label_w": 78
    },
    # Row 1 -> Row 2: 3 -> 4 (Down)
    {
        "x1": COL3_X + CARD_W / 2, "y1": ROW1_Y + CARD_H,
        "x2": COL3_X + CARD_W / 2, "y2": ROW2_Y,
        "type": "standard", "label": "PRECISION Θ", "label_w": 78
    },
    # Row 2: 4 -> 5 (Left)
    {
        "x1": COL3_X, "y1": ROW2_Y + CARD_H / 2,
        "x2": COL2_X + CARD_W, "y2": ROW2_Y + CARD_H / 2,
        "type": "novel", "label": "CAUSAL DAG W", "label_w": 82
    },
    # Row 2: 5 -> 6 (Left)
    {
        "x1": COL2_X, "y1": ROW2_Y + CARD_H / 2,
        "x2": COL1_X + CARD_W, "y2": ROW2_Y + CARD_H / 2,
        "type": "novel", "label": "VALIDATED W*", "label_w": 80
    },
    # Row 2 -> Row 3: 6 -> 7 (Down)
    {
        "x1": COL1_X + CARD_W / 2, "y1": ROW2_Y + CARD_H,
        "x2": COL1_X + CARD_W / 2, "y2": ROW3_Y,
        "type": "novel", "label": "DRIFT ALERTS", "label_w": 78
    },
    # Row 3: 7 -> 8 (Right)
    {
        "x1": COL1_X + CARD_W, "y1": ROW3_Y + CARD_H / 2,
        "x2": COL2_X, "y2": ROW3_Y + CARD_H / 2,
        "type": "novel", "label": "VOL FORECASTS", "label_w": 84
    },
    # Row 3: 8 -> 9 (Right)
    {
        "x1": COL2_X + CARD_W, "y1": ROW3_Y + CARD_H / 2,
        "x2": COL3_X, "y2": ROW3_Y + CARD_H / 2,
        "type": "standard", "label": "PORTFOLIO PnL", "label_w": 82
    }
]

def generate_svg_markup(is_standalone=True):
    svg_lines = []
    if is_standalone:
        svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" style="background:#f8f9fa;">')
    else:
        svg_lines.append(f'<svg id="main-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}">')

    # Defs
    svg_lines.append("""  <defs>
    <!-- Arrowhead markers -->
    <marker id="arr-slate" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
      <polygon points="0 0.5, 6 3.5, 0 6.5" fill="#475569" />
    </marker>
    <marker id="arr-coral" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
      <polygon points="0 0.5, 6 3.5, 0 6.5" fill="#ea580c" />
    </marker>
  </defs>""")

    # Outer Container
    svg_lines.append(f'  <!-- Outer Card Container -->')
    svg_lines.append(f'  <rect x="{CONTAINER_X}" y="{CONTAINER_Y}" width="{CONTAINER_W}" height="{CONTAINER_H}" rx="12" fill="#f8f9fa" stroke="#e2e8f0" stroke-width="1.5" />')

    # Header
    svg_lines.append("""  <!-- Header Block -->
  <text x="48" y="52" fill="#64748b" font-family="'Geist Mono', monospace" font-size="9" font-weight="600" letter-spacing="1.5">ARCHITECTURE FLOWCHART</text>
  <text x="48" y="88" fill="#0f172a" font-family="'Instrument Serif', Georgia, serif" font-size="32" font-weight="400">Causal Volatility Framework</text>
  <text x="48" y="112" fill="#475569" font-family="'Geist', -apple-system, BlinkMacSystemFont, sans-serif" font-size="13" font-weight="400">End-to-end pipeline from factor residualization to contagion-pruned risk execution.</text>""")

    # Connectors Layer (rendered behind nodes or with precise endpoints)
    svg_lines.append('  <!-- Connectors Layer -->')
    for conn in CONNECTORS:
        x1, y1 = conn["x1"], conn["y1"]
        x2, y2 = conn["x2"], conn["y2"]
        is_novel = conn["type"] == "novel"
        stroke_color = "#ea580c" if is_novel else "#475569"
        marker_id = "url(#arr-coral)" if is_novel else "url(#arr-slate)"
        text_color = "#ea580c" if is_novel else "#475569"
        badge_border = "#fed7aa" if is_novel else "#e2e8f0"

        # Arrow calculation with slight offset so arrowhead touches card perimeter
        if x1 < x2:  # Rightward
            line_x2 = x2 - 2
            line_y2 = y2
        elif x1 > x2:  # Leftward
            line_x2 = x2 + 2
            line_y2 = y2
        elif y1 < y2:  # Downward
            line_x2 = x2
            line_y2 = y2 - 2
        else:
            line_x2, line_y2 = x2, y2

        svg_lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{line_x2}" y2="{line_y2}" stroke="{stroke_color}" stroke-width="1.5" marker-end="{marker_id}" />')

        # Centered Masked Label
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        lbl_w = conn["label_w"]
        lbl_h = 18
        rx = mid_x - lbl_w / 2
        ry = mid_y - lbl_h / 2

        svg_lines.append(f'  <rect x="{rx:.1f}" y="{ry:.1f}" width="{lbl_w}" height="{lbl_h}" rx="3" fill="#f8f9fa" stroke="{badge_border}" stroke-width="0.8" />')
        svg_lines.append(f'  <text x="{mid_x:.1f}" y="{mid_y + 3.5:.1f}" text-anchor="middle" fill="{text_color}" font-family="\'Geist Mono\', monospace" font-size="8.5" font-weight="600" letter-spacing="0.4">{conn["label"]}</text>')

    # Nodes Layer
    svg_lines.append('  <!-- Nodes Layer -->')
    for node in NODES:
        nx, ny = node["x"], node["y"]
        is_novel = node["type"] == "novel"
        card_fill = "#fff7ed" if is_novel else "#ffffff"
        card_stroke = "#ea580c" if is_novel else "#334155"
        card_stroke_w = "1.5"
        
        tag_border = "#fed7aa" if is_novel else "#94a3b8"
        tag_color = "#ea580c" if is_novel else "#64748b"
        title_color = "#0f172a"
        sub_color = "#ea580c" if is_novel else "#64748b"

        tag_w = node["tag_w"]
        tag_x = nx + 12
        tag_y = ny + 10
        center_x = nx + CARD_W / 2

        # Card container group with tooltip attributes
        svg_lines.append(f'  <g class="node-group" data-id="{node["id"]}" tabindex="0">')
        # Main Card Box
        svg_lines.append(f'    <rect class="card-rect" x="{nx}" y="{ny}" width="{CARD_W}" height="{CARD_H}" rx="{CARD_R}" fill="{card_fill}" stroke="{card_stroke}" stroke-width="{card_stroke_w}" />')
        # Tag Badge
        svg_lines.append(f'    <rect x="{tag_x}" y="{tag_y}" width="{tag_w}" height="14" rx="3" fill="none" stroke="{tag_border}" stroke-width="1" />')
        svg_lines.append(f'    <text x="{tag_x + tag_w / 2:.1f}" y="{tag_y + 10:.1f}" text-anchor="middle" fill="{tag_color}" font-family="\'Geist Mono\', monospace" font-size="8" font-weight="600" letter-spacing="0.5">{node["tag"]}</text>')
        # Title
        svg_lines.append(f'    <text x="{center_x:.1f}" y="{ny + 44}" text-anchor="middle" fill="{title_color}" font-family="\'Geist\', -apple-system, BlinkMacSystemFont, sans-serif" font-size="13.5" font-weight="600">{node["title"]}</text>')
        # Sublabel
        svg_lines.append(f'    <text x="{center_x:.1f}" y="{ny + 62}" text-anchor="middle" fill="{sub_color}" font-family="\'Geist Mono\', monospace" font-size="9.5" font-weight="400">{node["sublabel"]}</text>')
        svg_lines.append('  </g>')

    # Bottom Separator Line & Legend
    svg_lines.append("""  <!-- Bottom Separator & Legend -->
  <line x1="48" y1="624" x2="892" y2="624" stroke="#e2e8f0" stroke-width="1" />
  
  <text x="48" y="658" fill="#64748b" font-family="'Geist Mono', monospace" font-size="9" font-weight="600" letter-spacing="1.5">LEGEND</text>
  
  <!-- Legend Item 1: Standard -->
  <rect x="120" y="647" width="14" height="14" rx="3" fill="#ffffff" stroke="#334155" stroke-width="1.4" />
  <text x="142" y="658" fill="#475569" font-family="'Geist', -apple-system, BlinkMacSystemFont, sans-serif" font-size="11" font-weight="400">Standard Pipeline Stage</text>
  
  <!-- Legend Item 2: Novel Causal Engine -->
  <rect x="290" y="647" width="14" height="14" rx="3" fill="#fff7ed" stroke="#ea580c" stroke-width="1.4" />
  <text x="312" y="658" fill="#ea580c" font-family="'Geist', -apple-system, BlinkMacSystemFont, sans-serif" font-size="11" font-weight="500">Novel Causal Engine</text>

  <!-- Capstone Annotation -->
  <text x="892" y="658" text-anchor="end" fill="#94a3b8" font-family="'Geist Mono', monospace" font-size="9.5">MScFE Capstone M2 Blueprint</text>
</svg>""")

    return "\n".join(svg_lines)

def build_interactive_html():
    svg_markup = generate_svg_markup(is_standalone=False)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Causal Volatility Framework — M2 Architecture Blueprint</title>
  
  <!-- Typography: Instrument Serif, Geist, Geist Mono -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600;700&family=Geist:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet" />
  
  <!-- FontAwesome 6 Icons -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" />

  <style>
    :root {{
      --bg: #f1f5f9;
      --surface: #ffffff;
      --surface-subtle: #f8fafc;
      --ink: #0f172a;
      --ink-muted: #475569;
      --border: #cbd5e1;
      --border-subtle: #e2e8f0;
      --coral: #ea580c;
      --coral-subtle: #fff7ed;
      --slate-stroke: #334155;
    }}

    [data-theme="dark"] {{
      --bg: #0b0f19;
      --surface: #111827;
      --surface-subtle: #1f2937;
      --ink: #f8fafc;
      --ink-muted: #94a3b8;
      --border: #334155;
      --border-subtle: #1e293b;
      --coral: #f97316;
      --coral-subtle: #431407;
      --slate-stroke: #94a3b8;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg);
      color: var(--ink);
      width: 100vw;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      user-select: none;
    }}

    /* Top Action Bar */
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--border-subtle);
      padding: 10px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 50;
      flex-shrink: 0;
    }}

    .header-title {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .header-title h1 {{
      font-family: 'Instrument Serif', Georgia, serif;
      font-size: 20px;
      font-weight: 400;
      color: var(--ink);
    }}

    .badge-m2 {{
      font-family: 'Geist Mono', monospace;
      font-size: 10px;
      font-weight: 600;
      background: var(--coral-subtle);
      color: var(--coral);
      padding: 2px 8px;
      border-radius: 9999px;
      border: 1px solid #fed7aa;
    }}

    .header-actions {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .btn {{
      font-family: 'Geist', sans-serif;
      font-size: 12px;
      font-weight: 500;
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--ink);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}

    .btn:hover {{
      background: var(--surface-subtle);
      border-color: var(--ink-muted);
    }}

    .btn-primary {{
      background: #2563eb;
      color: #ffffff;
      border-color: #2563eb;
    }}

    .btn-primary:hover {{
      background: #1d4ed8;
    }}

    /* Viewport Area */
    #viewport {{
      flex: 1;
      width: 100%;
      height: calc(100vh - 56px);
      position: relative;
      overflow: hidden;
      cursor: grab;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    #viewport.dragging {{
      cursor: grabbing;
    }}

    #diagram-container {{
      transform-origin: center center;
      will-change: transform;
      filter: drop-shadow(0 4px 12px rgba(0,0,0,0.06));
    }}

    svg {{
      display: block;
      width: {WIDTH}px;
      height: {HEIGHT}px;
    }}

    /* Node Hover & Interactive Tooltips */
    .node-group {{
      cursor: pointer;
      outline: none;
      transition: transform 0.15s ease;
    }}

    .node-group:hover .card-rect {{
      stroke-width: 2.2px;
      filter: drop-shadow(0 2px 8px rgba(0,0,0,0.08));
    }}

    /* Tooltip Floating Card */
    #tooltip {{
      position: absolute;
      display: none;
      background: #0f172a;
      color: #f8fafc;
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 12px;
      max-width: 280px;
      box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
      pointer-events: none;
      z-index: 100;
      border: 1px solid #334155;
      line-height: 1.4;
    }}

    #tooltip .tt-title {{
      font-family: 'Geist', sans-serif;
      font-weight: 600;
      color: #38bdf8;
      margin-bottom: 4px;
    }}

    #tooltip .tt-desc {{
      font-family: 'Geist', sans-serif;
      font-size: 11px;
      color: #cbd5e1;
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-title">
      <h1>Causal Volatility Framework</h1>
      <span class="badge-m2">M2 Methodology Flowchart</span>
    </div>
    <div class="header-actions">
      <button class="btn" id="btn-theme" title="Toggle Theme"><i class="fas fa-moon"></i> Theme</button>
      <button class="btn" id="btn-reset" title="Reset View"><i class="fas fa-compress-arrows-alt"></i> Reset</button>
      <button class="btn" id="btn-export-svg" title="Save Vector SVG"><i class="fas fa-file-code"></i> SVG</button>
      <button class="btn btn-primary" id="btn-export-png" title="Download High-Res PNG"><i class="fas fa-download"></i> Export PNG</button>
    </div>
  </header>

  <div id="viewport">
    <div id="diagram-container">
      {svg_markup}
    </div>
  </div>

  <div id="tooltip">
    <div class="tt-title" id="tt-title"></div>
    <div class="tt-desc" id="tt-desc"></div>
  </div>

  <script>
    // Node Tooltip Metadata
    const NODE_INFO = {{
{','.join(f'      "{n["id"]}": {{ title: "{n["tooltip_title"]}", desc: "{n["tooltip_desc"]}" }}' for n in NODES)}
    }};

    // Pan and Zoom Controller
    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let isDragging = false;
    let startX, startY;

    const viewport = document.getElementById('viewport');
    const container = document.getElementById('diagram-container');
    const tooltip = document.getElementById('tooltip');
    const ttTitle = document.getElementById('tt-title');
    const ttDesc = document.getElementById('tt-desc');

    function updateTransform() {{
      container.style.transform = `translate(${{translateX}}px, ${{translateY}}px) scale(${{scale}})`;
    }}

    viewport.addEventListener('wheel', (e) => {{
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      scale = Math.min(Math.max(0.4, scale * zoomFactor), 2.5);
      updateTransform();
    }});

    viewport.addEventListener('mousedown', (e) => {{
      if (e.target.closest('.header-actions') || e.target.closest('button')) return;
      isDragging = true;
      viewport.classList.add('dragging');
      startX = e.clientX - translateX;
      startY = e.clientY - translateY;
    }});

    window.addEventListener('mousemove', (e) => {{
      if (!isDragging) return;
      translateX = e.clientX - startX;
      translateY = e.clientY - startY;
      updateTransform();
    }});

    window.addEventListener('mouseup', () => {{
      isDragging = false;
      viewport.classList.remove('dragging');
    }});

    // Reset Zoom
    document.getElementById('btn-reset').addEventListener('click', () => {{
      scale = 1;
      translateX = 0;
      translateY = 0;
      updateTransform();
    }});

    // Dark Mode Toggle
    document.getElementById('btn-theme').addEventListener('click', () => {{
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      document.documentElement.setAttribute('data-theme', isDark ? 'light' : 'dark');
    }});

    // Tooltip Interactions
    document.querySelectorAll('.node-group').forEach(node => {{
      node.addEventListener('mouseenter', (e) => {{
        const id = node.getAttribute('data-id');
        const info = NODE_INFO[id];
        if (info) {{
          ttTitle.textContent = info.title;
          ttDesc.textContent = info.desc;
          tooltip.style.display = 'block';
        }}
      }});

      node.addEventListener('mousemove', (e) => {{
        tooltip.style.left = (e.clientX + 16) + 'px';
        tooltip.style.top = (e.clientY + 16) + 'px';
      }});

      node.addEventListener('mouseleave', () => {{
        tooltip.style.display = 'none';
      }});
    }});

    // Export SVG
    document.getElementById('btn-export-svg').addEventListener('click', () => {{
      const svgEl = document.getElementById('main-svg');
      const serializer = new XMLSerializer();
      let source = serializer.serializeToString(svgEl);
      if (!source.match(/^<svg[^>]+xmlns="http:\/\/www\.w3\.org\/2000\/svg"/)) {{
        source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
      }}
      const blob = new Blob([source], {{ type: 'image/svg+xml;charset=utf-8' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'm2_causal_architecture_simple.svg';
      a.click();
      URL.revokeObjectURL(url);
    }});

    // Export PNG
    document.getElementById('btn-export-png').addEventListener('click', () => {{
      const svgEl = document.getElementById('main-svg');
      const serializer = new XMLSerializer();
      const source = serializer.serializeToString(svgEl);
      const blob = new Blob([source], {{ type: 'image/svg+xml;charset=utf-8' }});
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {{
        const canvas = document.createElement('canvas');
        canvas.width = {WIDTH} * 2;
        canvas.height = {HEIGHT} * 2;
        const ctx = canvas.getContext('2d');
        ctx.scale(2, 2);
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, {WIDTH}, {HEIGHT});
        ctx.drawImage(img, 0, 0);
        const pngUrl = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = pngUrl;
        a.download = 'm2_causal_architecture_simple.png';
        a.click();
        URL.revokeObjectURL(url);
      }};
      img.src = url;
    }});
  </script>
</body>
</html>
"""
    return html

def main():
    print("Generating M2 Causal Architecture Flowchart (Type [A])...")
    
    # Generate Standalone SVG
    svg_content = generate_svg_markup(is_standalone=True)
    
    # Generate Interactive HTML
    html_content = build_interactive_html()

    target_dirs = [
        PRIMARY_DIR / "Flowcharts",
        ALT_DIR / "Flowcharts"
    ]

    for td in target_dirs:
        td.mkdir(parents=True, exist_ok=True)
        
        svg_path = td / "m2_causal_architecture_simple.svg"
        html_path = td / "m2_causal_architecture_simple.html"
        
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"✅ Wrote SVG & HTML to: {td}")

    # Render PNG using headless Chrome
    html_src = target_dirs[0] / "m2_causal_architecture_simple.html"
    png_out = target_dirs[0] / "m2_causal_architecture_simple.png"
    png_alt_out = target_dirs[1] / "m2_causal_architecture_simple.png"

    # Also render directly from SVG to ensure razor-sharp edges without browser chrome header
    # We can screenshot the standalone SVG or the HTML
    # Let's create a temporary minimalist HTML that embeds the standalone SVG without header
    temp_html = target_dirs[0] / "_render_temp.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600;700&family=Geist:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #f8f9fa; width: {WIDTH}px; height: {HEIGHT}px; overflow: hidden; }}
  </style>
</head>
<body>
{svg_content}
</body>
</html>""")

    cmd = [
        "/usr/bin/google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--window-size={WIDTH},{HEIGHT}",
        f"--screenshot={png_out}",
        str(temp_html)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        if temp_html.exists():
            temp_html.unlink()

        if png_out.exists() and png_out.stat().st_size > 5000:
            print(f"✅ Generated High-Resolution PNG: {png_out} ({png_out.stat().st_size} bytes)")
            # Copy to alt workspace
            with open(png_out, "rb") as rf, open(png_alt_out, "wb") as wf:
                wf.write(rf.read())
            print(f"✅ Synced PNG to: {png_alt_out}")
        else:
            print(f"⚠️ Chrome render issue: {res.stderr.decode()}")
    except Exception as e:
        print(f"⚠️ Error running chrome: {e}")

if __name__ == "__main__":
    main()
