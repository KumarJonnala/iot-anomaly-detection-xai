"""
IoT Anomaly Detection XAI Dashboard
Human-centred HITL system for IoT sensor anomaly review with natural language explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Health Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

SENSOR_COLS   = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]
SENSOR_LABELS = {
    "air_temp_k":     "Air Temperature",
    "process_temp_k": "Process Temperature",
    "rot_speed_rpm":  "Rotation Speed",
    "torque_nm":      "Torque",
    "tool_wear_min":  "Tool Wear",
}
SENSOR_UNITS = {
    "air_temp_k":     "K",
    "process_temp_k": "K",
    "rot_speed_rpm":  "rpm",
    "torque_nm":      "Nm",
    "tool_wear_min":  "min",
}
FAILURE_TYPES = ["HDF", "PWF", "OSF", "TWF", "RNF"]
FAILURE_NAMES = {
    "HDF": "Heat Dissipation",
    "PWF": "Power Failure",
    "OSF": "Overstrain",
    "TWF": "Tool Wear",
    "RNF": "Random Failure",
}
FAILURE_COLORS = {
    "NORMAL": "#0EA5E9",
    "HDF":    "#EF4444",
    "PWF":    "#F59E0B",
    "OSF":    "#10B981",
    "TWF":    "#8B5CF6",
    "RNF":    "#6B7280",
}
RULE_THRESHOLDS = {
    "hdf": {"max_temp_diff": 8.6, "max_rot_speed": 1380},
    "twf": {"min_tool_wear": 200},
    "osf": {"min_wear_torque": 11000},
}

# ── SVG icon set (stroke-based, Heroicons-style) ──────────────────────────────
def _svg(path_d: str, w: int = 18, extra: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{w}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" {extra}>'
        f'{path_d}</svg>'
    )

ICONS = {
    "overview": _svg(
        '<rect x="3" y="3" width="7" height="7" rx="1"/>'
        '<rect x="14" y="3" width="7" height="7" rx="1"/>'
        '<rect x="3" y="14" width="7" height="7" rx="1"/>'
        '<rect x="14" y="14" width="7" height="7" rx="1"/>'
    ),
    "detector": _svg(
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>'
        '<path d="M4.93 4.93a10 10 0 0 0 0 14.14"/>'
        '<path d="M16.24 7.76a6 6 0 0 1 0 8.49"/>'
        '<path d="M7.76 7.76a6 6 0 0 0 0 8.49"/>'
    ),
    "alert": _svg(
        '<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>'
        '<path d="M13.73 21a2 2 0 0 1-3.46 0"/>'
    ),
    "explain": _svg(
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        '<line x1="11" y1="8" x2="11" y2="11"/>'
        '<line x1="11" y1="14" x2="11.01" y2="14"/>'
    ),
    "accuracy": _svg(
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>'
        '<line x1="2" y1="20" x2="22" y2="20"/>'
    ),
    "sensor": _svg(
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
    ),
    "health": _svg(
        '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'
    ),
    "target": _svg(
        '<circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>'
    ),
    "warning": _svg(
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/>'
    ),
    "check": _svg(
        '<polyline points="20 6 9 17 4 12"/>'
    ),
    "settings": _svg(
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06'
        'a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09'
        'A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06'
        'A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09'
        'A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06'
        'A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09'
        'a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06'
        'A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09'
        'a1.65 1.65 0 0 0-1.51 1z"/>'
    ),
}

# ── Plotly light theme ────────────────────────────────────────────────────────
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#1E293B", size=12),
    xaxis=dict(
        gridcolor="#E2E8F0", linecolor="#CBD5E1", zerolinecolor="#E2E8F0",
        tickfont=dict(color="#1E293B", size=11),
        title_font=dict(color="#1E293B", size=12),
    ),
    yaxis=dict(
        gridcolor="#E2E8F0", linecolor="#CBD5E1", zerolinecolor="#E2E8F0",
        tickfont=dict(color="#1E293B", size=11),
        title_font=dict(color="#1E293B", size=12),
    ),
    legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor="#CBD5E1",
                font=dict(color="#1E293B", size=12)),
)

def pl(**overrides):
    """Merge chart overrides with the base theme. Dicts are merged one level deep
    so that e.g. xaxis=dict(range=[0,1]) preserves the base tickfont/title_font."""
    import copy
    out = copy.deepcopy(_PL)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


# ── CSS — light industry-standard theme ──────────────────────────────────────
st.markdown("""
<style>
/* ─────────────────────────────────────
   DESIGN TOKENS
───────────────────────────────────── */
:root {
  --bg:          #F8FAFC;
  --surface:     #FFFFFF;
  --raised:      #F1F5F9;
  --border:      #E2E8F0;
  --border-md:   #CBD5E1;

  /* All ink values hardened for crystal-clear contrast on white */
  --ink-1:  #0A0F1E;   /* near-black — headings, values */
  --ink-2:  #1E293B;   /* dark slate — body, labels */
  --ink-3:  #334155;   /* mid slate — secondary text, captions */
  --ink-4:  #475569;   /* softer — smallest labels (still ≥4.5:1 on white) */

  --accent:   #4338CA;  /* indigo-700 — selected nav, active accent */
  --danger:   #DC2626;
  --warning:  #B45309;
  --success:  #047857;
  --info:     #0369A1;

  --danger-bg:  #FEF2F2;
  --warning-bg: #FFFBEB;
  --success-bg: #ECFDF5;
  --info-bg:    #F0F9FF;
  --accent-bg:  #EEF2FF;

  --radius:    10px;
  --radius-lg: 14px;
  --shadow-sm: 0 1px 3px rgba(15,23,42,.07), 0 1px 2px rgba(15,23,42,.05);
  --shadow:    0 4px 12px rgba(15,23,42,.10), 0 1px 3px rgba(15,23,42,.06);
  --ease: cubic-bezier(.16,1,.3,1);
}

/* ─────────────────────────────────────
   GLOBAL
───────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
}
[data-testid="block-container"] {
  padding-top: 1.8rem !important;
  max-width: 1300px !important;
}
[data-testid="stMarkdownContainer"] p {
  color: var(--ink-2) !important;
  line-height: 1.65 !important;
}
[data-testid="stMarkdownContainer"] strong { color: var(--ink-1) !important; }
.stCaption { color: var(--ink-3) !important; font-size: 0.80rem !important; }
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* ─────────────────────────────────────
   SIDEBAR
───────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1.5px solid var(--border) !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--border) !important;
  margin: 10px 0 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stCaption {
  color: var(--ink-3) !important;
  font-size: 0.81rem !important;
}
[data-testid="stSidebar"] a { color: var(--accent) !important; }

/* ── Nav — bordered tab buttons ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
  gap: 5px !important;
  flex-direction: column !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  display: flex !important;
  align-items: center !important;
  width: 100% !important;
  padding: 9px 13px !important;
  border-radius: 8px !important;
  border: 1.5px solid #CBD5E1 !important;
  background: #FFFFFF !important;
  cursor: pointer !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  box-shadow: 0 1px 2px rgba(15,23,42,.05) !important;
  transition: background .12s ease, border-color .12s ease !important;
  margin: 0 !important;
}
/* Unselected text — very dark */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] label span,
[data-testid="stSidebar"] [data-testid="stRadio"] label div {
  color: #1E293B !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: #F1F5F9 !important;
  border-color: #94A3B8 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover span,
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover div {
  color: #0A0F1E !important;
}
/* Active / selected — solid indigo fill */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: #4338CA !important;
  border-color: #4338CA !important;
  box-shadow: 0 2px 6px rgba(67,56,202,.30) !important;
  font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) div,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * {
  color: #FFFFFF !important;
  background: transparent !important;    /* don't re-colour children's bg */
}
/* Re-apply bg only to the label itself */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: #4338CA !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
  position: absolute !important; opacity: 0 !important;
  width: 0 !important; height: 0 !important;
}

/* ── Tooltip popup — white text on dark bg ── */
/* Target every possible Streamlit/Baseweb tooltip selector */
div[role="tooltip"],
div[role="tooltip"] *,
[data-baseweb="tooltip"],
[data-baseweb="tooltip"] > div,
[data-baseweb="tooltip"] div,
[data-baseweb="tooltip"] p,
[data-baseweb="tooltip"] span,
[data-testid="stTooltipContent"],
[data-testid="stTooltipContent"] * {
  color: #FFFFFF !important;
}
div[role="tooltip"],
[data-baseweb="tooltip"] > div {
  background-color: #1E293B !important;
  border-radius: 6px !important;
}
/* (?) icon itself */
[data-testid="stTooltipIcon"] svg,
[data-testid="stTooltipIcon"] svg path {
  stroke: #334155 !important;
  color: #334155 !important;
}

/* ── Progress bar text label ── */
[data-testid="stProgress"] p,
[data-testid="stProgress"] > div > p {
  color: #0A0F1E !important;
  font-weight: 600 !important;
  font-size: 0.83rem !important;
}

/* ── Force Plotly SVG axis tick & title text dark ──
   Belt-and-suspenders: CSS on the SVG element wins over JS layout props
   when there's a theming conflict.                                        */
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text {
  fill: #0A0F1E !important;
  font-size: 11px !important;
}
.js-plotly-plot .plotly .g-xtitle text,
.js-plotly-plot .plotly .g-ytitle text {
  fill: #0A0F1E !important;
  font-weight: 600 !important;
}
.js-plotly-plot .plotly .gtitle {
  fill: #0A0F1E !important;
}
.js-plotly-plot .plotly .legend text {
  fill: #1E293B !important;
}

/* ── Number input (stepper) ── */
[data-testid="stSidebar"] [data-testid="stNumberInput"] > label > div > p {
  color: var(--ink-2) !important;
  font-size: 0.81rem !important;
  font-weight: 700 !important;
}
[data-testid="stNumberInput"] [data-baseweb="input"] {
  border: 1.5px solid var(--border-md) !important;
  border-radius: var(--radius) !important;
  background: var(--surface) !important;
  overflow: hidden !important;
}
[data-testid="stNumberInput"] input {
  background: var(--surface) !important;
  color: var(--ink-1) !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  text-align: center !important;
}
[data-testid="stNumberInput"] button {
  background: var(--raised) !important;
  border-color: var(--border) !important;
  color: var(--ink-2) !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
}
[data-testid="stNumberInput"] button:hover {
  background: var(--border) !important;
  color: var(--ink-1) !important;
}

/* Checkbox */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
  color: var(--ink-3) !important;
  font-size: 0.82rem !important;
}

/* ─────────────────────────────────────
   METRIC CARDS
───────────────────────────────────── */
div[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 16px 18px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: box-shadow .2s var(--ease) !important;
}
div[data-testid="metric-container"]:hover {
  box-shadow: var(--shadow) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
  font-size: 0.73rem !important; font-weight: 700 !important;
  letter-spacing: .07em !important; text-transform: uppercase !important;
  color: var(--ink-3) !important;   /* was ink-4 — now clearly readable */
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-size: 1.85rem !important; font-weight: 800 !important;
  letter-spacing: -.03em !important; color: var(--ink-1) !important;
  line-height: 1.15 !important;
}

/* ─────────────────────────────────────
   TABS
───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--raised) !important;
  border-radius: var(--radius) !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1.5px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 7px !important;
  color: var(--ink-3) !important;
  font-size: 0.83rem !important;
  font-weight: 500 !important;
  padding: 6px 14px !important;
  background: transparent !important;
}
.stTabs [aria-selected="true"][data-baseweb="tab"] {
  background: var(--surface) !important;
  color: var(--ink-1) !important;
  font-weight: 600 !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ─────────────────────────────────────
   EXPANDERS
───────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  margin-bottom: 8px !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stExpander"] summary {
  padding: 13px 16px !important;
  font-weight: 600 !important;
  font-size: 0.90rem !important;
  color: var(--ink-1) !important;
}
[data-testid="stExpander"] summary:hover {
  background: var(--raised) !important;
}

/* ─────────────────────────────────────
   SELECTBOX
───────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  background: var(--surface) !important;
  border-color: var(--border-md) !important;
  border-radius: var(--radius) !important;
  color: var(--ink-1) !important;
}

/* Sidebar page-picker selectbox */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
  border: 1.5px solid #CBD5E1 !important;
  border-radius: 8px !important;
  background: #FFFFFF !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  color: #1E293B !important;
  box-shadow: 0 1px 2px rgba(15,23,42,.05) !important;
  padding: 2px 6px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:hover {
  border-color: #94A3B8 !important;
  background: #F1F5F9 !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div span {
  color: #1E293B !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
}

/* ─────────────────────────────────────
   DATAFRAME
───────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ─────────────────────────────────────
   ALERTS
───────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius-lg) !important;
  border-width: 1.5px !important;
}

/* ─────────────────────────────────────
   PROGRESS
───────────────────────────────────── */
[data-testid="stProgress"] > div > div { background: var(--accent) !important; }

/* ─────────────────────────────────────
   CUSTOM COMPONENTS
───────────────────────────────────── */

/* KPI card */
.kpi-card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
  height: 100%;
}
.kpi-card.danger  { border-top: 3px solid var(--danger);  }
.kpi-card.warning { border-top: 3px solid var(--warning); }
.kpi-card.success { border-top: 3px solid var(--success); }
.kpi-card.info    { border-top: 3px solid var(--info);    }
.kpi-card.neutral { border-top: 3px solid var(--border-md); }
.kpi-card.accent  { border-top: 3px solid var(--accent);  }
.kpi-label {
  font-size: 0.70rem; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; color: var(--ink-3); margin-bottom: 5px;
}
.kpi-value {
  font-size: 2rem; font-weight: 800; letter-spacing: -.03em;
  color: var(--ink-1); line-height: 1;
}
.kpi-sub { font-size: 0.79rem; color: var(--ink-2); margin-top: 5px; font-weight: 500; }

/* Model card */
.model-card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px;
  box-shadow: var(--shadow-sm);
  height: 100%;
}
.model-icon  { margin-bottom: 10px; color: var(--ink-2); }
.model-name  { font-size: 0.95rem; font-weight: 700; color: var(--ink-1); margin-bottom: 6px; }
.model-desc  { font-size: 0.83rem; color: var(--ink-2); line-height: 1.55; }
.model-on  {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.72rem; font-weight: 700; color: var(--success);
  background: var(--success-bg); padding: 3px 10px;
  border-radius: 20px; margin-top: 12px;
}
.model-off {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.72rem; font-weight: 700; color: var(--ink-4);
  background: var(--raised); padding: 3px 10px;
  border-radius: 20px; margin-top: 12px;
}

/* Sensor row */
.sensor-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 9px 14px; border-radius: 8px; margin-bottom: 5px;
}
.sensor-row.bad  { background: var(--danger-bg);  border: 1px solid #FECACA; }
.sensor-row.warn { background: var(--warning-bg); border: 1px solid #FDE68A; }
.sensor-row.good { background: var(--raised);      border: 1px solid var(--border); }
.sensor-label   { font-size: 0.87rem; font-weight: 700; color: var(--ink-1); }
.sensor-reading { font-size: 0.87rem; font-family: ui-monospace,monospace; color: var(--ink-1); font-weight: 600; }
.sensor-tag     { font-size: 0.72rem; font-weight: 700; padding: 2px 9px; border-radius: 20px; }
.sensor-tag.bad  { color: #7F1D1D; background: #FEE2E2; }
.sensor-tag.warn { color: #78350F; background: #FEF3C7; }
.sensor-tag.good { color: var(--ink-3); background: var(--raised); border: 1px solid var(--border-md); }

/* Page header */
.page-header { margin-bottom: 28px; }
.page-title {
  font-size: clamp(1.5rem, 2.5vw, 2.1rem);
  font-weight: 800; letter-spacing: -.03em;
  color: var(--ink-1); line-height: 1.15;
  text-wrap: balance; margin-bottom: 6px;
}
.page-subtitle { font-size: 0.90rem; color: var(--ink-2); line-height: 1.60; max-width: 72ch; font-weight: 400; }

/* Section heading */
.sec-head {
  display: flex; align-items: center; gap: 8px;
  font-size: 1.05rem; font-weight: 700; color: var(--ink-1);
  letter-spacing: -.015em; margin-bottom: 4px;
}
.sec-head svg { color: var(--ink-2); flex-shrink: 0; }
.sec-note { font-size: 0.83rem; color: var(--ink-3); margin-bottom: 16px; margin-left: 26px; font-weight: 400; }

/* RQ progress card */
.rq-card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
}
.rq-title { font-size: 0.90rem; font-weight: 700; color: var(--ink-1); margin-bottom: 5px; }
.rq-sub   { font-size: 0.82rem; color: var(--ink-2); margin-bottom: 12px; line-height: 1.55; }

/* Metadata summary card */
.meta-card {
  background: var(--raised);
  border: 1.5px solid var(--border-md);
  border-radius: var(--radius-lg);
  padding: 16px 18px;
}
.meta-row { margin-bottom: 10px; }
.meta-label { font-size: 0.70rem; font-weight: 700; text-transform: uppercase;
              letter-spacing: .07em; color: var(--ink-3); }
.meta-value { font-size: 0.88rem; font-weight: 600; color: var(--ink-1); margin-top: 2px; }

/* Floating note */
.note-box {
  background: var(--accent-bg);
  border: 1.5px solid #C7D2FE;
  border-radius: var(--radius);
  padding: 12px 16px;
  font-size: 0.82rem; color: var(--ink-2);
  margin-top: 14px;
}

/* Sidebar brand header */
.brand {
  padding: 2px 0 12px;
}
.brand-name {
  font-size: 1rem; font-weight: 800; color: var(--ink-1); letter-spacing: -.025em;
}
.brand-sub {
  font-size: 0.78rem; color: var(--ink-3); margin-top: 3px; font-weight: 500;
}

/* Section label inside sidebar */
.sb-section {
  font-size: 0.72rem; font-weight: 700; letter-spacing: .07em;
  text-transform: uppercase; color: var(--ink-3); margin-bottom: 8px;
  display: flex; align-items: center; gap: 5px;
}

/* ─────────────────────────────────────
   ENTRANCE ANIMATIONS
   Elegant fade + gentle upward drift.
   Staggered per column so cards arrive
   in a smooth cascade rather than all
   at once.
───────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0);    }
}

/* Page header */
.page-header {
  animation: fadeUp 0.56s cubic-bezier(0.16, 1, 0.3, 1) both;
}

/* Section headings — slight offset after header */
.sec-head {
  animation: fadeUp 0.53s cubic-bezier(0.16, 1, 0.3, 1) 0.08s both;
}
.sec-note {
  animation: fadeUp 0.53s cubic-bezier(0.16, 1, 0.3, 1) 0.14s both;
}

/* Custom KPI cards — staggered by column position */
.kpi-card {
  animation: fadeUp 0.59s cubic-bezier(0.16, 1, 0.3, 1) both;
}
[data-testid="stColumn"]:nth-child(1) .kpi-card { animation-delay:   0ms; }
[data-testid="stColumn"]:nth-child(2) .kpi-card { animation-delay:  84ms; }
[data-testid="stColumn"]:nth-child(3) .kpi-card { animation-delay: 168ms; }
[data-testid="stColumn"]:nth-child(4) .kpi-card { animation-delay: 252ms; }
[data-testid="stColumn"]:nth-child(5) .kpi-card { animation-delay: 336ms; }
[data-testid="stColumn"]:nth-child(6) .kpi-card { animation-delay: 420ms; }

/* Streamlit native metric cards — same stagger */
div[data-testid="metric-container"] {
  animation: fadeUp 0.59s cubic-bezier(0.16, 1, 0.3, 1) both;
}
[data-testid="stColumn"]:nth-child(1) div[data-testid="metric-container"] { animation-delay:   0ms; }
[data-testid="stColumn"]:nth-child(2) div[data-testid="metric-container"] { animation-delay:  84ms; }
[data-testid="stColumn"]:nth-child(3) div[data-testid="metric-container"] { animation-delay: 168ms; }
[data-testid="stColumn"]:nth-child(4) div[data-testid="metric-container"] { animation-delay: 252ms; }
[data-testid="stColumn"]:nth-child(5) div[data-testid="metric-container"] { animation-delay: 336ms; }
[data-testid="stColumn"]:nth-child(6) div[data-testid="metric-container"] { animation-delay: 420ms; }

/* Model cards */
.model-card {
  animation: fadeUp 0.63s cubic-bezier(0.16, 1, 0.3, 1) both;
}
[data-testid="stColumn"]:nth-child(1) .model-card { animation-delay:   0ms; }
[data-testid="stColumn"]:nth-child(2) .model-card { animation-delay: 105ms; }
[data-testid="stColumn"]:nth-child(3) .model-card { animation-delay: 210ms; }
[data-testid="stColumn"]:nth-child(4) .model-card { animation-delay: 315ms; }

/* Plotly charts — brief hold then rise */
[data-testid="stPlotlyChart"] {
  animation: fadeUp 0.70s cubic-bezier(0.16, 1, 0.3, 1) 0.11s both;
}

/* Dataframes / HTML tables */
[data-testid="stDataFrame"],
[data-testid="stMarkdownContainer"] table {
  animation: fadeUp 0.63s cubic-bezier(0.16, 1, 0.3, 1) 0.08s both;
}

/* Alert queue expanders — stagger first 10 */
[data-testid="stExpander"] {
  animation: fadeUp 0.50s cubic-bezier(0.16, 1, 0.3, 1) both;
}
[data-testid="stExpander"]:nth-child(1)  { animation-delay:  28ms; }
[data-testid="stExpander"]:nth-child(2)  { animation-delay:  77ms; }
[data-testid="stExpander"]:nth-child(3)  { animation-delay: 126ms; }
[data-testid="stExpander"]:nth-child(4)  { animation-delay: 175ms; }
[data-testid="stExpander"]:nth-child(5)  { animation-delay: 224ms; }
[data-testid="stExpander"]:nth-child(6)  { animation-delay: 273ms; }
[data-testid="stExpander"]:nth-child(7)  { animation-delay: 322ms; }
[data-testid="stExpander"]:nth-child(8)  { animation-delay: 371ms; }
[data-testid="stExpander"]:nth-child(9)  { animation-delay: 420ms; }
[data-testid="stExpander"]:nth-child(10) { animation-delay: 469ms; }

/* Info / warning boxes */
[data-testid="stAlert"] {
  animation: fadeUp 0.59s cubic-bezier(0.16, 1, 0.3, 1) 0.14s both;
}

/* Progress bars */
[data-testid="stProgress"] {
  animation: fadeUp 0.53s cubic-bezier(0.16, 1, 0.3, 1) 0.11s both;
}

/* Tabs strip */
.stTabs [data-baseweb="tab-list"] {
  animation: fadeUp 0.50s cubic-bezier(0.16, 1, 0.3, 1) 0.06s both;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
</style>
""", unsafe_allow_html=True)


# ── UI helpers ────────────────────────────────────────────────────────────────
def kpi(col, label: str, value: str, sub: str = "", variant: str = "neutral"):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="kpi-card {variant}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    sub = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="page-header"><div class="page-title">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def section(icon_key: str, title: str, note: str = ""):
    note_html = f'<div class="sec-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="sec-head">{ICONS.get(icon_key,"")}<span>{title}</span></div>'
        f'{note_html}',
        unsafe_allow_html=True,
    )


def model_card(col, icon_key: str, name: str, desc: str, enabled: bool):
    status = ('<div class="model-on">● Active</div>' if enabled
              else '<div class="model-off">○ Disabled</div>')
    col.markdown(
        f'<div class="model-card">'
        f'<div class="model-icon">{ICONS.get(icon_key,"")}</div>'
        f'<div class="model-name">{name}</div>'
        f'<div class="model-desc">{desc}</div>'
        f'{status}</div>',
        unsafe_allow_html=True,
    )


def sensor_row_html(label: str, value: float, unit: str, z: float, thr: float) -> str:
    if abs(z) > thr:
        cls, tag_cls, tag = "bad",  "bad",  "Abnormal"
    elif abs(z) > 2.0:
        cls, tag_cls, tag = "warn", "warn", "Elevated"
    else:
        cls, tag_cls, tag = "good", "good", "Normal"
    direction = "high" if z > 0 else "low"
    sub = f'<div style="font-size:.74rem;color:#475569;font-weight:500">unusually {direction}</div>' if abs(z) > 2.0 else ""
    return (
        f'<div class="sensor-row {cls}">'
        f'<div><div class="sensor-label">{label}</div>{sub}</div>'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div class="sensor-reading">{value:.2f}&nbsp;{unit}</div>'
        f'<div class="sensor-tag {tag_cls}">{tag}</div>'
        f'</div></div>'
    )


# ── Autoencoder ───────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        def __init__(self, n_features: int = 5, latent_dim: int = 4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, n_features * 2), nn.ReLU(),
                nn.BatchNorm1d(n_features * 2),
                nn.Linear(n_features * 2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, n_features * 2), nn.ReLU(),
                nn.BatchNorm1d(n_features * 2),
                nn.Linear(n_features * 2, n_features),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))


# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(zscore_thresh, if_contamination, fusion_thresh, ae_percentile, use_ae):
    df = pd.read_csv(DATA_DIR / "ai4i_clean.csv")
    with open(DATA_DIR / "ai4i_ranges.json") as f:
        ranges = json.load(f)

    df_orig = df.copy()
    for col in SENSOR_COLS:
        r = ranges[col]
        df_orig[col] = df[col] * (r["max"] - r["min"]) + r["min"]

    df_orig["temp_diff_k"] = df_orig["process_temp_k"] - df_orig["air_temp_k"]
    df_orig["wear_torque"]  = df_orig["tool_wear_min"] * df_orig["torque_nm"]

    t = RULE_THRESHOLDS
    df["rule_hdf"] = (
        (df_orig["temp_diff_k"] < t["hdf"]["max_temp_diff"]) &
        (df_orig["rot_speed_rpm"] < t["hdf"]["max_rot_speed"])
    )
    df["rule_twf"] = df_orig["tool_wear_min"] >= t["twf"]["min_tool_wear"]
    df["rule_osf"] = df_orig["wear_torque"] > t["osf"]["min_wear_torque"]
    df["temp_diff_k"] = df_orig["temp_diff_k"]
    df["wear_torque"]  = df_orig["wear_torque"]

    for col in SENSOR_COLS:
        g_mean, g_std = df[col].mean(), df[col].std()
        df[f"{col}_zscore_global"]  = (df[col] - g_mean) / (g_std + 1e-9)
        r_mean = df[col].rolling(50, min_periods=5).mean()
        r_std  = df[col].rolling(50, min_periods=5).std()
        df[f"{col}_zscore_dynamic"] = (df[col] - r_mean) / (r_std + 1e-9)

    df["zscore_max"] = df[
        [f"{c}_zscore_global"  for c in SENSOR_COLS] +
        [f"{c}_zscore_dynamic" for c in SENSOR_COLS]
    ].abs().fillna(0).max(axis=1)
    df["zscore_flag"] = df["zscore_max"] > zscore_thresh

    X   = df[SENSOR_COLS].values
    clf = IsolationForest(contamination=if_contamination, n_estimators=200, random_state=42)
    clf.fit(X)
    raw = clf.score_samples(X)
    df["if_score"] = 1 - (raw - raw.min()) / (raw.max() - raw.min())
    df["if_flag"]  = clf.predict(X) == -1

    ae_threshold = None
    if use_ae and TORCH_AVAILABLE:
        X_all    = df[SENSOR_COLS].values.astype(np.float32)
        X_normal = X_all[df["machine_failure"].values == 0]
        n_val    = int(len(X_normal) * 0.1)
        X_t      = torch.FloatTensor(X_normal[n_val:])
        X_v      = torch.FloatTensor(X_normal[:n_val])
        loader   = DataLoader(TensorDataset(X_t), batch_size=256, shuffle=True)
        model    = Autoencoder(5, 4)
        opt      = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        crit     = nn.MSELoss()
        best_val, no_improve, best_state = float("inf"), 0, None
        for _ in range(80):
            model.train()
            for (batch,) in loader:
                loss = crit(model(batch), batch)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = crit(model(X_v), X_v).item()
            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= 10:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            X_recon = model(torch.FloatTensor(X_all)).numpy()
        ae_per_sensor = (X_all - X_recon) ** 2
        ae_errors     = ae_per_sensor.mean(axis=1)
        normal_errors = ae_errors[df["machine_failure"].values == 0]
        ae_threshold  = float(np.percentile(normal_errors, ae_percentile))
        df["ae_error"] = ae_errors
        df["ae_score"] = np.clip(ae_errors / (ae_threshold * 2), 0, 1)
        df["ae_flag"]  = ae_errors > ae_threshold
        for i, col in enumerate(SENSOR_COLS):
            df[f"ae_error_{col}"] = ae_per_sensor[:, i]

    zscore_norm = np.clip(df["zscore_max"] / 5.0, 0, 1)
    has_ae      = "ae_score" in df.columns
    combined    = (
        0.30 * zscore_norm + 0.40 * df["if_score"] + 0.30 * df["ae_score"]
        if has_ae else
        0.43 * zscore_norm + 0.57 * df["if_score"]
    )
    df["zscore_norm"]    = zscore_norm
    df["combined_score"] = combined

    rule_any     = df["rule_hdf"] | df["rule_twf"] | df["rule_osf"]
    df["anomaly"] = (combined > fusion_thresh) | rule_any

    flags   = [df["zscore_flag"].astype(int), df["if_flag"].astype(int)]
    if has_ae:
        flags.append(df["ae_flag"].astype(int))
    n_agree   = sum(flags)
    agree_map = {0: "none", 1: "one_only", 2: "two_of_three", 3: "all_three"}
    df["agreement"] = n_agree.map(agree_map)

    return df, ranges, ae_threshold


def denorm(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    out = df.copy()
    for col in SENSOR_COLS:
        r = ranges[col]
        out[col] = df[col] * (r["max"] - r["min"]) + r["min"]
    return out


# ── Explanation generator ─────────────────────────────────────────────────────
def generate_explanation(row: pd.Series, ranges: dict) -> str:
    parts: list[str] = []

    if row.get("rule_hdf", False):
        td = row.get("temp_diff_k", float("nan"))
        parts.append(
            f"**Heat Dissipation Issue Detected** — The temperature gap between the process "
            f"and surrounding air is only **{td:.1f} K** (safe minimum: 8.6 K), while the "
            f"machine is also running below 1,380 rpm. Together, these conditions mean the "
            f"machine cannot shed heat fast enough — thermal damage risk is elevated."
        )
    if row.get("rule_twf", False):
        parts.append(
            "**Tool Wear Limit Reached** — The cutting tool has been in use for 200+ minutes. "
            "This is the recommended replacement point. Continuing past this threshold "
            "significantly raises the risk of sudden tool breakage and workpiece damage."
        )
    if row.get("rule_osf", False):
        wt = row.get("wear_torque", float("nan"))
        parts.append(
            f"**Mechanical Overload Risk** — The wear-torque product is **{wt:,.0f} min·Nm** "
            f"(limit: 11,000). The tool is under excessive mechanical stress relative to its "
            f"wear state — failure risk is high."
        )

    if not parts:
        z         = row.get("zscore_max", 0)
        sensor_zs = {c: abs(float(row.get(f"{c}_zscore_global", 0))) for c in SENSOR_COLS}
        worst     = max(sensor_zs, key=sensor_zs.get)
        wz        = sensor_zs[worst]
        if z > 3.0:
            parts.append(
                f"**Unusual Reading Detected** — {SENSOR_LABELS[worst]} is "
                f"**{wz:.1f} standard deviations** away from its normal range. "
                f"This reading is highly unusual compared to normal operation."
            )

    agree     = row.get("agreement", "none")
    conf_note = {
        "all_three":    "**High Confidence** — All three detection methods flagged this reading independently. This is a strong signal that something is genuinely wrong.",
        "two_of_three": "**Medium Confidence** — Two out of three methods agree. Worth investigating promptly.",
        "one_only":     "**Low Confidence** — Only one method flagged this. It may be a false alarm — review carefully before acting.",
        "none":         "**Rule-based flag only** — Flagged by an engineering threshold, not by the statistical models.",
    }.get(agree, "")
    if conf_note:
        parts.append(conf_note)

    ft = row.get("failure_type", "NORMAL")
    if ft != "NORMAL":
        parts.append(
            f"**Confirmed Failure (Ground Truth):** This reading is labelled as a "
            f"**{FAILURE_NAMES.get(ft, ft)}** in the dataset — confirms the system caught a real event."
        )

    parts.append(
        "**Recommended Action:** Check the maintenance log for recent service on this machine. "
        "If any rule-based flag is present, schedule preventive maintenance before the next "
        "production run. Use the sensor timeline to assess whether this is a spike or a trend."
    )
    return "\n\n".join(parts)


def severity(score: float) -> str:
    if score >= 0.7: return "HIGH"
    if score >= 0.5: return "MEDIUM"
    return "LOW"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand">'
        '<div class="brand-name">Machine Health Monitor</div>'
        '<div class="brand-sub">IoT anomaly detection · AI explanations</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        f'<div class="sb-section">{ICONS["overview"]}  Page</div>',
        unsafe_allow_html=True,
    )
    page = st.selectbox(
        "Page",
        ["Overview", "How It Works", "Alert Queue"],
        label_visibility="collapsed",
        key="ds_page",
    )

    st.markdown("---")
    st.markdown(
        f'<div class="sb-section">{ICONS["settings"]}  Detection Controls</div>',
        unsafe_allow_html=True,
    )

    zscore_thresh = st.number_input(
        "Statistical threshold (σ)", min_value=2.0, max_value=5.0,
        value=3.0, step=0.1, format="%.1f",
        help="How many standard deviations away from the mean before flagging. 3σ is the scientific convention.",
    )
    if_contamination = st.number_input(
        "Expected fault rate", min_value=0.010, max_value=0.100,
        value=0.034, step=0.001, format="%.3f",
        help="Approximate fraction of readings that are real faults. Set close to your machine's actual failure rate.",
    )
    fusion_thresh = st.number_input(
        "Alert threshold", min_value=0.30, max_value=0.80,
        value=0.50, step=0.05, format="%.2f",
        help="Combined score needed to raise an alert. Higher = fewer, more confident alerts.",
    )
    ae_percentile = st.number_input(
        "Neural net sensitivity", min_value=90, max_value=99,
        value=95, step=1,
        help="Percentile of reconstruction error above which the neural network raises a flag.",
    )
    use_ae = st.checkbox(
        "Enable Neural Network (~30 s first run)",
        value=TORCH_AVAILABLE,
        help=(
            "Adds a third detection method. Results are cached after the first run."
            if TORCH_AVAILABLE else "PyTorch not installed — run: pip install torch"
        ),
    )
    st.markdown("---")
    st.caption("Dataset: AI4I 2020 Predictive Maintenance")
    st.caption("UCI ML Repository · [Source](https://doi.org/10.24432/C5HS5C)")


# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("Analysing sensor data… (cached after first run)"):
    df, ranges, ae_threshold = run_pipeline(
        zscore_thresh, if_contamination, fusion_thresh, ae_percentile, use_ae
    )

anomaly_df = df[df["anomaly"]].copy()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    page_header(
        "Fleet Overview",
        "10,000 sensor readings from the AI4I 2020 predictive maintenance dataset — "
        "5 sensors, 5 failure modes, one detection system.",
    )

    tp = int((df["anomaly"] & (df["machine_failure"] == 1)).sum())
    fp = int((df["anomaly"] & (df["machine_failure"] == 0)).sum())
    fn = int((~df["anomaly"] & (df["machine_failure"] == 1)).sum())
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, "Total Readings",  f"{len(df):,}",             "all sensor snapshots",          "neutral")
    kpi(c2, "Real Failures",   f"{df['machine_failure'].sum():,}", f"{df['machine_failure'].mean():.1%} of all readings", "danger")
    kpi(c3, "Alerts Raised",   f"{df['anomaly'].sum():,}",  "by the combined system",        "warning")
    kpi(c4, "Failures Caught", f"{tp:,}",                   f"recall {recall:.0%}",           "success")
    kpi(c5, "False Alarms",    f"{fp:,}",                   f"precision {precision:.0%}",     "danger")
    kpi(c6, "Missed Failures", f"{fn:,}",                   "not caught",                    "warning")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Failure distribution ────────────────────────────────────────────────
    section("sensor", "Failure Type Distribution",
            "How often each failure mode occurred across all 10,000 readings.")

    ft_counts = df["failure_type"].value_counts().reset_index()
    ft_counts.columns = ["Failure Type", "Count"]

    col_bar, col_pie = st.columns([3, 2])

    with col_bar:
        fig_bar = px.bar(
            ft_counts, x="Failure Type", y="Count",
            color="Failure Type", color_discrete_map=FAILURE_COLORS,
            text="Count",
        )
        fig_bar.update_traces(textposition="outside", textfont_color="#1E293B")
        fig_bar.update_layout(**pl(
            height=340, showlegend=False,
            margin=dict(t=10, b=0),
            xaxis_title="", yaxis_title="Number of Readings",
        ))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        # Pie: breakdown of failure types only (NORMAL excluded so all slices
        # are visible; NORMAL count shown as callout below)
        n_normal  = int((df["failure_type"] == "NORMAL").sum())
        n_failure = int((df["failure_type"] != "NORMAL").sum())

        ft_fail = df[df["failure_type"] != "NORMAL"]["failure_type"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=[FAILURE_NAMES.get(k, k) for k in ft_fail.index],
            values=ft_fail.values,
            hole=0.52,
            marker=dict(colors=[FAILURE_COLORS[k] for k in ft_fail.index],
                        line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            textposition="outside",
            automargin=True,
        ))
        fig_pie.update_layout(**pl(
            height=340,
            margin=dict(t=10, b=40, l=20, r=20),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{n_failure}</b><br><span style='font-size:10px'>failures</span>",
                x=0.5, y=0.5, font_size=14, showarrow=False,
                font_color="#0F172A",
            )],
        ))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(
            f"Pie shows breakdown of the **{n_failure} failure readings** only. "
            f"The remaining **{n_normal:,}** readings were normal operation."
        )

    st.markdown("---")

    # ── Sensor distributions ────────────────────────────────────────────────
    section("health", "Sensor Readings — Normal vs Failure",
            "Compare each sensor during normal operation (blue) vs failure events (red). "
            "Non-overlapping peaks indicate sensors that reliably predict failures.")

    df_orig = denorm(df, ranges)
    tabs    = st.tabs([f"{SENSOR_LABELS[c]}  ({SENSOR_UNITS[c]})" for c in SENSOR_COLS])
    for tab, col in zip(tabs, SENSOR_COLS):
        with tab:
            left, right = st.columns([3, 1])
            with left:
                r    = ranges[col]
                fig3 = go.Figure()
                fig3.add_trace(go.Histogram(
                    x=df_orig.loc[df_orig["machine_failure"] == 0, col],
                    name="Normal", marker_color="#0EA5E9", opacity=0.65, nbinsx=50,
                ))
                fig3.add_trace(go.Histogram(
                    x=df_orig.loc[df_orig["machine_failure"] == 1, col],
                    name="Failure", marker_color="#EF4444", opacity=0.7, nbinsx=50,
                ))
                fig3.add_vrect(
                    x0=r["mean"] - 2 * r["std"], x1=r["mean"] + 2 * r["std"],
                    fillcolor="#0EA5E9", opacity=0.06, line_width=0,
                    annotation_text="Normal range (±2σ)",
                    annotation_font_size=10, annotation_font_color="#1E293B",
                )
                fig3.update_layout(**pl(
                    barmode="overlay", height=280,
                    xaxis_title=f"{SENSOR_LABELS[col]} ({SENSOR_UNITS[col]})",
                    legend=dict(orientation="h", y=1.16),
                    margin=dict(t=10, b=0),
                ))
                st.plotly_chart(fig3, use_container_width=True)
            with right:
                r = ranges[col]
                st.markdown(
                    f'<div style="background:#F8FAFC;border:1.5px solid #E2E8F0;'
                    f'border-radius:10px;padding:14px;margin-top:10px">'
                    f'<div class="kpi-label">Average</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:#0F172A;margin-bottom:10px">'
                    f'{r["mean"]:.2f} {SENSOR_UNITS[col]}</div>'
                    f'<div class="kpi-label">Std Dev</div>'
                    f'<div style="font-size:1rem;color:#334155;margin-bottom:10px">{r["std"]:.2f}</div>'
                    f'<div class="kpi-label">Min</div>'
                    f'<div style="font-size:.95rem;color:#334155;margin-bottom:8px">{r["min"]:.1f}</div>'
                    f'<div class="kpi-label">Max</div>'
                    f'<div style="font-size:.95rem;color:#334155">{r["max"]:.1f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Correlation matrix ──────────────────────────────────────────────────
    section("target", "Sensor Correlation Matrix",
            "Values near +1 (dark red): sensors move together. Near −1 (dark blue): they move in "
            "opposite directions. Strong correlations may indicate mechanical coupling between components.")

    corr    = df_orig[SENSOR_COLS].corr()
    fig_cor = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        x=[SENSOR_LABELS[c] for c in SENSOR_COLS],
        y=[SENSOR_LABELS[c] for c in SENSOR_COLS],
    )
    fig_cor.update_layout(**pl(height=380, margin=dict(t=10, b=0)))
    st.plotly_chart(fig_cor, use_container_width=True)

    st.markdown("---")

    section("warning", "Historical Failure Log",
            "Every reading where a confirmed machine failure was recorded in the dataset.")
    past = df_orig.loc[df["machine_failure"] == 1, ["failure_type"] + SENSOR_COLS].copy()
    past = past.rename(columns={**SENSOR_LABELS, "failure_type": "Failure Type"})
    past.index.name = "Reading #"
    st.dataframe(
        past.style.format({SENSOR_LABELS[c]: "{:.2f}" for c in SENSOR_COLS}),
        use_container_width=True, height=320,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "How It Works":
    page_header(
        "How the Detection System Works",
        "Three independent statistical methods plus engineering rules monitor each sensor reading. "
        "When multiple methods agree, confidence is higher and so is the alert priority.",
    )

    section("detector", "Detection Methods",
            "Each method analyses the data from a different angle — they complement each other.")
    st.markdown("<br>", unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    model_card(mc1, "accuracy", "Statistical Z-Score",
               "Compares each sensor reading against thousands of historical normal readings. "
               "Flags values that fall unusually far from the expected range.",
               enabled=True)
    model_card(mc2, "sensor", "Isolation Forest",
               "Looks at all five sensors simultaneously and identifies readings that are "
               "unusual as a combination — even if each sensor individually looks fine.",
               enabled=True)
    model_card(mc3, "health", "Neural Network",
               "Trains on fault-free readings to learn what 'normal' looks like, then "
               "flags any reading it struggles to reconstruct accurately.",
               enabled="ae_flag" in df.columns)
    model_card(mc4, "settings", "Engineering Rules",
               "Hard thresholds set by domain experts — e.g. tool in use over 200 min, "
               "or temperature differential dangerously low at low speed.",
               enabled=True)

    st.markdown(
        '<div class="note-box">'
        '<strong>Final decision:</strong> An alert is raised when the combined score from all '
        'active methods exceeds the Alert Threshold, OR when any engineering rule is triggered — '
        'whichever happens first. Adjust these thresholds using the controls in the sidebar.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    section("target", "Current Detection Settings",
            "These are the values currently active. Change them in the sidebar.")
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    kpi(cfg1, "Statistical Threshold", f"{zscore_thresh:.1f} σ",
        "readings beyond this are flagged", "info")
    kpi(cfg2, "Expected Fault Rate",   f"{if_contamination:.1%}",
        "tunes Isolation Forest sensitivity", "info")
    kpi(cfg3, "Alert Threshold",       f"{fusion_thresh:.2f}",
        "combined score to trigger alert", "info")
    kpi(cfg4, "Neural Net Sensitivity",
        f"p{ae_percentile}" if "ae_flag" in df.columns else "Disabled",
        "top percentile of reconstruction error", "info")

    st.markdown("---")

    section("alert", f"Anomalies Found — {len(anomaly_df):,} of {len(df):,} readings",
            "Sorted by severity — highest combined score first.")

    if len(anomaly_df) == 0:
        st.info("No anomalies detected. Try lowering the Alert Threshold in the sidebar.")
    else:
        rows = []
        for ri, row in anomaly_df.sort_values("combined_score", ascending=False).iterrows():
            triggers = []
            if row.get("rule_hdf"): triggers.append("Heat dissipation rule")
            if row.get("rule_twf"): triggers.append("Tool wear rule")
            if row.get("rule_osf"): triggers.append("Overstrain rule")
            if row.get("zscore_flag"): triggers.append("Z-Score")
            if row.get("if_flag"):    triggers.append("Isolation Forest")
            if row.get("ae_flag", False): triggers.append("Neural Network")
            rows.append({
                "Reading #":   ri,
                "Severity":    severity(row["combined_score"]),
                "Score":       f"{row['combined_score']:.3f}",
                "Flagged by":  ", ".join(triggers) or "—",
                "True Label":  "Failure" if row["machine_failure"] == 1 else "Normal",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ALERT QUEUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Alert Queue":
    page_header(
        "Alert Queue",
        "All active alerts sorted by urgency. Expand an alert to see which sensors triggered it.",
    )

    n_total = len(anomaly_df)
    n_high  = int((anomaly_df["combined_score"] >= 0.7).sum()) if n_total else 0
    n_med   = int(((anomaly_df["combined_score"] >= 0.5) & (anomaly_df["combined_score"] < 0.7)).sum()) if n_total else 0
    n_low   = n_total - n_high - n_med

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Alerts",   str(n_total), "currently active",           "accent")
    kpi(c2, "High Priority",  str(n_high),  "score ≥ 0.70 — act quickly", "danger")
    kpi(c3, "Medium Priority",str(n_med),   "score 0.50–0.69 — review",   "warning")
    kpi(c4, "Low Priority",   str(n_low),   "score < 0.50 — monitor",     "success")

    st.markdown("---")

    if n_total == 0:
        st.info("No active alerts. Lower the Alert Threshold in the sidebar to see more.")
        st.stop()

    sort_col = st.selectbox(
        "Sort by",
        ["combined_score", "zscore_max", "if_score"],
        format_func=lambda c: {
            "combined_score": "Severity (highest first)",
            "zscore_max":     "Statistical deviation",
            "if_score":       "Isolation Forest score",
        }[c],
    )
    view_df = anomaly_df.sort_values(sort_col, ascending=False)
    df_orig = denorm(df, ranges)
    shown   = min(len(view_df), 60)
    st.caption(f"Showing {shown} of {n_total} alerts")

    for row_idx, row in view_df.head(60).iterrows():
        sev      = severity(row["combined_score"])
        sev_icon = {"HIGH": "●", "MEDIUM": "●", "LOW": "●"}[sev]
        sev_col  = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}[sev]
        rule_flags = [
            name for flag, name in [
                ("rule_hdf", "Heat Dissipation"), ("rule_twf", "Tool Wear"), ("rule_osf", "Overstrain"),
            ] if row.get(flag)
        ]
        true_label = "Confirmed Failure" if row["machine_failure"] == 1 else "Normal in ground truth"
        header = (
            f'<span style="color:{sev_col}">{sev_icon}</span> '
            f'Reading #{row_idx:,}  ·  {sev} severity  ·  Score {row["combined_score"]:.3f}  ·  {true_label}'
        )

        with st.expander(
            f"Reading #{row_idx:,}  ·  {sev}  ·  {true_label}",
            expanded=False,
        ):
            if rule_flags:
                st.markdown(
                    f'<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;'
                    f'padding:10px 14px;font-size:.84rem;color:#991B1B;margin-bottom:12px">'
                    f'Engineering rule triggered: <strong>{", ".join(rule_flags)}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div style="font-size:.74rem;font-weight:700;text-transform:uppercase;'
                'letter-spacing:.07em;color:#334155;margin-bottom:8px">Sensor Readings</div>',
                unsafe_allow_html=True,
            )
            rows_html = ""
            shown_any = False
            for c in SENSOR_COLS:
                z_signed = float(row.get(f"{c}_zscore_global", 0))
                z        = abs(z_signed)
                val_o    = float(df_orig.loc[row_idx, c])
                if z > 2.0:
                    shown_any = True
                    rows_html += sensor_row_html(SENSOR_LABELS[c], val_o, SENSOR_UNITS[c], z_signed, zscore_thresh)

            if shown_any:
                st.markdown(rows_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="font-size:.83rem;color:#334155;padding:8px 0">'
                    'All individual sensors are within range — this alert was raised by the '
                    'combined model score detecting an unusual pattern across sensors.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div style="margin-top:10px;font-size:.80rem;color:#334155">'
                f'Detector agreement: {row.get("agreement","—").replace("_"," ")}'
                f'</div>',
                unsafe_allow_html=True,
            )
