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
from sklearn.metrics import classification_report, confusion_matrix
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

    page = st.radio(
        "Navigation",
        ["Overview", "How It Works", "Alert Queue", "Explain an Alert", "Accuracy Report"],
        label_visibility="collapsed",
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
            "Sorted by severity. Use the Explain an Alert page to inspect any row in detail.")

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
                f'Detector agreement: {row.get("agreement","—").replace("_"," ")} · '
                f'Open <em>Explain an Alert</em> for a full explanation of Reading #{row_idx:,}.'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — XAI EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Explain an Alert":
    page_header(
        "Alert Explainer",
        "Select an alert to see a plain-English explanation of why it was flagged, "
        "which sensors were out of range, and what to do next.",
    )

    if len(anomaly_df) == 0:
        st.warning("No anomalies detected. Adjust the Detection Controls in the sidebar.")
        st.stop()

    anomaly_indices = anomaly_df.sort_values("combined_score", ascending=False).index.tolist()
    sel = st.selectbox(
        "Choose an alert to inspect",
        anomaly_indices,
        format_func=lambda i: (
            f"Reading #{i:,}  ·  Score {anomaly_df.loc[i,'combined_score']:.3f}  ·  "
            f"{severity(anomaly_df.loc[i,'combined_score'])} severity  ·  "
            f"{'Confirmed Failure' if anomaly_df.loc[i,'machine_failure']==1 else 'Normal in data'}"
        ),
    )

    row      = df.loc[sel]
    df_orig  = denorm(df, ranges)
    row_orig = df_orig.loc[sel]

    WIN   = 20
    start = max(0, sel - WIN)
    end   = min(len(df), sel + WIN + 1)
    ctx      = df.iloc[start:end]
    ctx_orig = df_orig.iloc[start:end]
    ctx_idx  = ctx.index

    st.markdown("---")
    section("sensor", f"Sensor Timeline — Reading #{sel:,} in context",
            f"±{WIN} readings around this alert. The red star marks the flagged reading. "
            "Check whether this is an isolated spike or part of a longer trend.")

    c_left, c_right = st.columns([3, 2])

    with c_left:
        ctx_sensor = st.selectbox(
            "Sensor to display",
            SENSOR_COLS,
            format_func=lambda c: f"{SENSOR_LABELS[c]} ({SENSOR_UNITS[c]})",
            key="xai_sensor",
        )
        r       = ranges[ctx_sensor]
        fig_ctx = go.Figure()
        fig_ctx.add_hrect(
            y0=r["mean"] - 2 * r["std"], y1=r["mean"] + 2 * r["std"],
            fillcolor="#0EA5E9", opacity=0.06, line_width=0,
            annotation_text="Normal band (±2σ)",
            annotation_font_size=10, annotation_font_color="#1E293B",
        )
        fig_ctx.add_hline(y=r["mean"], line_dash="dot", line_color="#0EA5E9",
                           line_width=1, opacity=0.4)
        fig_ctx.add_trace(go.Scatter(
            x=ctx_idx, y=ctx_orig[ctx_sensor], mode="lines+markers",
            line=dict(color="#0EA5E9", width=1.5), marker=dict(size=4),
            name=SENSOR_LABELS[ctx_sensor],
        ))
        fig_ctx.add_trace(go.Scatter(
            x=[sel], y=[row_orig[ctx_sensor]], mode="markers",
            marker=dict(color="#EF4444", size=16, symbol="star",
                        line=dict(color="white", width=1.5)),
            name=f"Alert (Reading #{sel})",
        ))
        other = ctx_idx[ctx["anomaly"] & (ctx_idx != sel)]
        if len(other) > 0:
            fig_ctx.add_trace(go.Scatter(
                x=other, y=ctx_orig.loc[other, ctx_sensor], mode="markers",
                marker=dict(color="#F59E0B", size=8, symbol="diamond"),
                name="Other nearby alerts", opacity=0.85,
            ))
        fig_ctx.update_layout(**pl(
            height=300,
            yaxis_title=f"{SENSOR_LABELS[ctx_sensor]} ({SENSOR_UNITS[ctx_sensor]})",
            xaxis_title="Reading number",
            legend=dict(orientation="h", y=1.18),
            margin=dict(t=10, b=10),
        ))
        st.plotly_chart(fig_ctx, use_container_width=True)

    with c_right:
        st.markdown(
            '<div style="font-size:.74rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:.07em;color:#334155;margin-bottom:10px">All Sensors at This Reading</div>',
            unsafe_allow_html=True,
        )
        rows_html = ""
        for c in SENSOR_COLS:
            r   = ranges[c]
            val = float(row_orig[c])
            z   = (val - r["mean"]) / (r["std"] + 1e-9)
            rows_html += sensor_row_html(SENSOR_LABELS[c], val, SENSOR_UNITS[c], z, zscore_thresh)
        st.markdown(rows_html, unsafe_allow_html=True)

        wv = ctx_orig[ctx_sensor].values
        st.markdown(
            f'<div style="margin-top:14px;background:#F8FAFC;border:1.5px solid #E2E8F0;'
            f'border-radius:10px;padding:12px 14px">'
            f'<div class="kpi-label">Window stats — {SENSOR_LABELS[ctx_sensor]}</div>'
            f'<div style="font-size:.83rem;color:#334155;margin-top:6px;line-height:1.9">'
            f'Median: <strong style="color:#0F172A">{np.median(wv):.3f} {SENSOR_UNITS[ctx_sensor]}</strong><br>'
            f'Std Dev: <strong style="color:#0F172A">{np.std(wv):.3f}</strong><br>'
            f'Range: <strong style="color:#0F172A">{np.min(wv):.3f} – {np.max(wv):.3f}</strong>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    section("accuracy", "Detection Method Scores",
            "How strongly each method flagged this reading. Bars past the dashed line triggered an alert.")
    d_left, d_right = st.columns(2)

    with d_left:
        det = {
            "Z-Score (Statistical)": float(row.get("zscore_norm", 0)),
            "Isolation Forest":       float(row.get("if_score", 0)),
        }
        if "ae_score" in row.index:
            det["Neural Network"] = float(row.get("ae_score", 0))
        det["Combined Score"] = float(row.get("combined_score", 0))

        clrs = ["#0EA5E9", "#10B981", "#8B5CF6", "#F59E0B"]
        fig_det = go.Figure(go.Bar(
            x=list(det.values()), y=list(det.keys()), orientation="h",
            marker=dict(color=clrs[:len(det)]),
            text=[f"{v:.3f}" for v in det.values()], textposition="outside",
            textfont=dict(color="#1E293B"),
        ))
        fig_det.add_vline(x=fusion_thresh, line_dash="dash", line_color="#EF4444",
                           annotation_text=f"Threshold ({fusion_thresh})",
                           annotation_font_color="#1E293B")
        fig_det.update_layout(**pl(height=250, xaxis=dict(range=[0, 1.18]),
                                   margin=dict(t=5, b=5, l=160)))
        st.plotly_chart(fig_det, use_container_width=True)

    with d_right:
        if all(f"ae_error_{c}" in row.index for c in SENSOR_COLS):
            section("health", "Neural Net — Per-Sensor Error",
                    "Higher = harder to reconstruct = more unusual.")
            ae_err = {SENSOR_LABELS[c]: float(row[f"ae_error_{c}"]) for c in SENSOR_COLS}
            fig_ae = go.Figure(go.Bar(
                x=list(ae_err.values()), y=list(ae_err.keys()), orientation="h",
                marker_color="#8B5CF6",
                text=[f"{v:.5f}" for v in ae_err.values()], textposition="outside",
                textfont=dict(color="#1E293B"),
            ))
            fig_ae.update_layout(**pl(height=250, margin=dict(t=5, b=5, l=160)))
            st.plotly_chart(fig_ae, use_container_width=True)
        else:
            section("sensor", "Z-Score per Sensor",
                    "Standard deviations each sensor is from its average normal value.")
            z_attr = {
                SENSOR_LABELS[c]: abs(float(row.get(f"{c}_zscore_global", 0)))
                for c in SENSOR_COLS
            }
            fig_z = go.Figure(go.Bar(
                x=list(z_attr.values()), y=list(z_attr.keys()), orientation="h",
                marker_color="#0EA5E9",
                text=[f"{v:.2f}σ" for v in z_attr.values()], textposition="outside",
                textfont=dict(color="#1E293B"),
            ))
            fig_z.add_vline(x=zscore_thresh, line_dash="dash", line_color="#EF4444",
                             annotation_text=f"Threshold ({zscore_thresh}σ)",
                             annotation_font_color="#1E293B")
            fig_z.update_layout(**pl(height=250, margin=dict(t=5, b=5, l=160)))
            st.plotly_chart(fig_z, use_container_width=True)

    st.markdown("---")

    section("explain", "Plain-English Explanation",
            "What the system found, what it means, and what you should do.")
    ex_col, meta_col = st.columns([3, 1])

    with ex_col:
        st.info(generate_explanation(row, ranges))

    with meta_col:
        sev_str   = severity(float(row["combined_score"]))
        sev_color = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}[sev_str]
        st.markdown(
            f'<div class="meta-card">'
            f'<div style="font-size:.70rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.08em;color:#334155;margin-bottom:12px">Alert Summary</div>'
            f'<div class="meta-row"><div class="meta-label">Severity</div>'
            f'<div class="meta-value" style="color:{sev_color}">{sev_str}</div></div>'
            f'<div class="meta-row"><div class="meta-label">Reading</div>'
            f'<div class="meta-value">#{sel:,}</div></div>'
            f'<div class="meta-row"><div class="meta-label">Score</div>'
            f'<div class="meta-value">{row["combined_score"]:.4f}</div></div>'
            f'<div class="meta-row"><div class="meta-label">Agreement</div>'
            f'<div class="meta-value">{row.get("agreement","N/A").replace("_"," ")}</div></div>'
            f'<div class="meta-row"><div class="meta-label">True Label</div>'
            f'<div class="meta-value" style="color:{"#EF4444" if row["machine_failure"]==1 else "#10B981"}">'
            f'{"Failure" if row["machine_failure"]==1 else "Normal"}</div></div>'
            f'<div class="meta-row"><div class="meta-label">Failure Type</div>'
            f'<div class="meta-value">{row.get("failure_type","NORMAL")}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    section("target", "Sensor Profile Radar",
            "Red shape: this reading's sensor values (0–1 normalised). "
            "Blue dashed: typical normal reading. Outward spikes show which sensors deviate most.")

    row_norm  = [float(row[c]) for c in SENSOR_COLS]
    glob_mean = [0.5] * len(SENSOR_COLS)
    labels    = [SENSOR_LABELS[c] for c in SENSOR_COLS] + [SENSOR_LABELS[SENSOR_COLS[0]]]

    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatterpolar(
        r=row_norm + [row_norm[0]], theta=labels,
        fill="toself", name=f"Reading #{sel}", line_color="#EF4444",
        fillcolor="rgba(239,68,68,0.12)",
    ))
    fig_rad.add_trace(go.Scatterpolar(
        r=glob_mean + [glob_mean[0]], theta=labels,
        fill="toself", name="Typical normal", line_color="#0EA5E9",
        line_dash="dash", fillcolor="rgba(14,165,233,0.05)",
    ))
    fig_rad.update_layout(**pl(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#E2E8F0", color="#334155"),
            angularaxis=dict(gridcolor="#E2E8F0", color="#475569"),
        ),
        height=380, margin=dict(t=30, b=30),
    ))
    st.plotly_chart(fig_rad, use_container_width=True)

    st.markdown(
        '<div style="background:#F8FAFC;border:1.5px solid #E2E8F0;border-radius:10px;'
        'padding:13px 16px;font-size:.80rem;color:#334155;margin-top:-8px">'
        '<strong style="color:#334155">Research note (Stage 3 — upcoming):</strong> '
        'This template explanation will be replaced by a locally-hosted language model '
        '(Mistral / Llama) evaluated with three prompting strategies — zero-shot, contextualised, '
        'and RAG — via an LLM-as-Judge pipeline (RQ2). Per-sensor neural network reconstruction '
        'error serves as the attribution signal.'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ACCURACY REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Accuracy Report":
    page_header(
        "Accuracy Report",
        "How well does each detection method identify real failures? "
        "Compared against ground-truth labels from the AI4I dataset.",
    )

    y_true    = df["machine_failure"]
    detectors = [("Z-Score", "zscore_flag"), ("Isolation Forest", "if_flag"), ("Combined System", "anomaly")]
    if "ae_flag" in df.columns:
        detectors.insert(2, ("Neural Network", "ae_flag"))

    rows_m = []
    for name, col in detectors:
        y_pred = df[col].astype(int)
        rep    = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        f      = rep.get("1", {})
        rows_m.append({
            "Method":        name,
            "Precision":     round(f.get("precision", 0), 3),
            "Recall":        round(f.get("recall",    0), 3),
            "F1 Score":      round(f.get("f1-score",  0), 3),
            "Alerts Raised": int(df[col].sum()),
            "False Alarm %": f"{(df[col] & (y_true==0)).sum() / (y_true==0).sum():.2%}",
        })

    section("accuracy", "Detection Quality per Method",
            "Precision: of all alerts raised, how many were real failures? "
            "Recall: of all real failures, how many were caught? "
            "F1: balanced score — higher is better (max = 1.0).")
    met_df = pd.DataFrame(rows_m)

    # Best value per numeric column (for highlighting)
    best = {c: met_df[c].max() for c in ["Precision", "Recall", "F1 Score"]}

    th_style = (
        "padding:11px 16px;text-align:left;background:#1E293B;color:#FFFFFF;"
        "font-size:.78rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;"
        "border-bottom:2px solid #334155;"
    )
    tbl = (
        '<table style="width:100%;border-collapse:collapse;font-size:.88rem;'
        'border:1.5px solid #E2E8F0;border-radius:12px;overflow:hidden">'
        "<thead><tr>"
        + "".join(f'<th style="{th_style}">{c}</th>' for c in met_df.columns)
        + "</tr></thead><tbody>"
    )
    for _, row in met_df.iterrows():
        is_combined = row["Method"] == "Combined System"
        row_bg  = "#EEF2FF" if is_combined else "#FFFFFF"
        row_bdr = "2px solid #818CF8" if is_combined else "1px solid #E2E8F0"
        tbl += f'<tr style="background:{row_bg};border-bottom:{row_bdr}">'
        for col in met_df.columns:
            val     = row[col]
            is_best = col in best and val == best[col]
            cell_bg = "#D1FAE5" if is_best else "transparent"
            weight  = "700" if col == "Method" or is_combined else "500"
            txt_col = "#0A0F1E"
            fmt_val = (
                f"{val:.3f}" if col in ("Precision", "Recall", "F1 Score")
                else f"{val:,}" if col == "Alerts Raised"
                else str(val)
            )
            tbl += (
                f'<td style="padding:11px 16px;color:{txt_col};font-weight:{weight};'
                f'background:{cell_bg}">{fmt_val}</td>'
            )
        tbl += "</tr>"
    tbl += "</tbody></table>"
    st.markdown(tbl, unsafe_allow_html=True)

    st.markdown("---")

    section("target", "Confusion Matrices",
            "Top-left: correctly called Normal. Bottom-right: correctly caught a failure. "
            "Top-right: false alarms. Bottom-left: failures the method missed.")
    cm_cols = st.columns(len(detectors))
    for cm_col, (name, col) in zip(cm_cols, detectors):
        with cm_col:
            cm     = confusion_matrix(y_true, df[col].astype(int))
            fig_cm = px.imshow(
                cm, text_auto=True,
                color_continuous_scale=[[0, "#F8FAFC"], [0.5, "#BFDBFE"], [1, "#1D4ED8"]],
                x=["Predicted: Normal", "Predicted: Failure"],
                y=["Actually: Normal",  "Actually: Failure"],
            )
            fig_cm.update_layout(**pl(
                title=dict(text=name, font=dict(size=13, color="#0F172A")),
                height=270, margin=dict(t=35, b=5, l=80, r=5),
                coloraxis_showscale=False,
            ))
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    section("warning", "Detection Rate per Failure Type",
            "Of each failure type, what fraction did the combined system successfully catch?")
    fr_rows = []
    for ftype in FAILURE_TYPES:
        mask  = df[ftype] == 1
        total = int(mask.sum())
        if total == 0:
            continue
        det = int(df.loc[mask, "anomaly"].sum())
        fr_rows.append({
            "Failure Type": FAILURE_NAMES[ftype],
            "Caught":       det,
            "Total":        total,
            "Hit Rate":     det / total,
        })
    fr_df = pd.DataFrame(fr_rows)
    fc1, fc2 = st.columns(2)
    with fc1:
        st.dataframe(
            fr_df.style.format({"Hit Rate": "{:.1%}", "Caught": "{:,}", "Total": "{:,}"}),
            use_container_width=True, hide_index=True,
        )
    with fc2:
        fig_fr = px.bar(
            fr_df, x="Failure Type", y="Hit Rate", color="Failure Type",
            color_discrete_map={FAILURE_NAMES[k]: v for k, v in FAILURE_COLORS.items() if k in FAILURE_NAMES},
            text=fr_df["Hit Rate"].map("{:.1%}".format), range_y=[0, 1.15],
        )
        fig_fr.update_traces(textposition="outside", textfont_color="#1E293B")
        fig_fr.update_layout(**pl(showlegend=False, height=300, margin=dict(t=5, b=0)))
        st.plotly_chart(fig_fr, use_container_width=True)

    st.markdown("---")

    section("health", "Combined Score Distribution — Normal vs Failure",
            "Blue = normal readings. Red = real failures. Good separation means the system reliably "
            "tells them apart. The dashed line is the current alert threshold.")
    fig_sd = go.Figure()
    fig_sd.add_trace(go.Histogram(
        x=df.loc[df["machine_failure"] == 0, "combined_score"],
        name="Normal readings", marker_color="#0EA5E9", opacity=0.65,
        nbinsx=60, histnorm="probability",
    ))
    fig_sd.add_trace(go.Histogram(
        x=df.loc[df["machine_failure"] == 1, "combined_score"],
        name="Real failures", marker_color="#EF4444", opacity=0.75,
        nbinsx=60, histnorm="probability",
    ))
    fig_sd.add_vline(
        x=fusion_thresh, line_dash="dash", line_color="#F59E0B",
        annotation_text=f"Alert threshold ({fusion_thresh})",
        annotation_font_color="#1E293B",
    )
    fig_sd.update_layout(**pl(
        barmode="overlay", height=310,
        xaxis_title="Combined Anomaly Score  (0 = normal  ·  1 = highly abnormal)",
        yaxis_title="Proportion of readings",
        legend=dict(orientation="h", y=1.14),
        margin=dict(t=10, b=10),
    ))
    st.plotly_chart(fig_sd, use_container_width=True)

    st.markdown("---")

    section("target", "Research Progress",
            "Status of the three research questions this dashboard supports.")
    rq1, rq2, rq3 = st.columns(3)

    _rq_card = (
        'background:#FFFFFF;border:1.5px solid #E2E8F0;border-radius:12px;padding:18px 20px;'
        'box-shadow:0 1px 3px rgba(15,23,42,.06);margin-bottom:10px'
    )
    _rq_title = 'font-size:.90rem;font-weight:700;color:#0A0F1E;margin-bottom:6px'
    _rq_sub   = 'font-size:.82rem;color:#1E293B;line-height:1.55;font-weight:400'
    _rq_prog  = 'font-size:.82rem;font-weight:700;color:#0A0F1E;margin-bottom:6px;margin-top:4px'

    with rq1:
        st.markdown(
            f'<div style="{_rq_card}">'
            f'<div style="{_rq_title}">RQ1 — Does AI explanation build trust?</div>'
            f'<div style="{_rq_sub}">Comparing operator trust (Likert scale) with vs without '
            f'AI-generated explanations.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<p style="{_rq_prog}">Progress: 20% — HITL review interface built</p>',
                    unsafe_allow_html=True)
        st.progress(0.20)

    with rq2:
        st.markdown(
            f'<div style="{_rq_card}">'
            f'<div style="{_rq_title}">RQ2 — Which prompting strategy works best?</div>'
            f'<div style="{_rq_sub}">Zero-shot, contextualised, and RAG prompting evaluated by an '
            f'LLM-as-Judge pipeline.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<p style="{_rq_prog}">Progress: 10% — Template explanation in place</p>',
                    unsafe_allow_html=True)
        st.progress(0.10)

    with rq3:
        st.markdown(
            f'<div style="{_rq_card}">'
            f'<div style="{_rq_title}">RQ3 — Is the dashboard usable?</div>'
            f'<div style="{_rq_sub}">System Usability Scale (SUS) survey after the user study. '
            f'Target: SUS ≥ 70 (Good).</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<p style="{_rq_prog}">Progress: 10% — Dashboard scaffold complete</p>',
                    unsafe_allow_html=True)
        st.progress(0.10)
