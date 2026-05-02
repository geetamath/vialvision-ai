import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from collections import defaultdict

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VialVision AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;900&display=swap');

:root {
    --bg: #0a0e1a;
    --panel: #0f1628;
    --border: #1e3a5f;
    --accent: #00d4ff;
    --green: #00ff88;
    --orange: #ff8c00;
    --red: #ff3366;
    --yellow: #ffe066;
    --text: #c8d8e8;
    --dim: #e8f4ff;
}

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}

/* Header Banner */
.header-banner {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1f3c 50%, #0a0e1a 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), var(--green), transparent);
}
.header-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 0;
}
.header-sub {
    font-size: 0.85rem;
    color: #c8d8e8;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.header-badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Share Tech Mono', monospace;
    margin-top: 0.5rem;
    margin-right: 0.5rem;
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin: 1rem 0;
}
.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.vial::after { background: var(--accent); }
.metric-card.sealed::after { background: var(--green); }
.metric-card.unsealed::after { background: var(--orange); }
.metric-card.Damaged::after { background: var(--red); }

.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-card.vial .metric-value { color: var(--accent); }
.metric-card.sealed .metric-value { color: var(--green); }
.metric-card.unsealed .metric-value { color: var(--orange); }
.metric-card.Damaged .metric-value { color: var(--red); }
.metric-label {
    font-size: 0.7rem;
    color: #a0b0c8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
}

/* Panel */
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    color: var(--text);  /* ADDED THIS LINE */
}
.panel-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Detection Tags */
.det-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    margin: 0.2rem;
}
.det-tag.vial { background: rgba(0,212,255,0.15); border: 1px solid var(--accent); color: var(--accent); }
.det-tag.sealed { background: rgba(0,255,136,0.15); border: 1px solid var(--green); color: var(--green); }
.det-tag.unsealed { background: rgba(255,140,0,0.15); border: 1px solid var(--orange); color: var(--orange); }
.det-tag.Damaged { background: rgba(255,51,102,0.15); border: 1px solid var(--red); color: var(--red); }

/* Progress bar */
.conf-bar-wrap { margin: 0.3rem 0; }
.conf-label { font-size: 0.75rem; color: #c8d8e8; display: flex; justify-content: space-between; }
.conf-bar { height: 4px; background: var(--border); border-radius: 2px; margin-top: 2px; }
.conf-fill { height: 100%; border-radius: 2px; transition: width 0.5s ease; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,212,255,0.05)) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.3) !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.3) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    color: #00d4ff !important;
    padding: 0.7rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* File uploader */
.stFileUploader {
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    background: rgba(0,212,255,0.02) !important;
}

/* Alerts */
.status-ok {
    background: rgba(0,100,60,0.6);
    border: 1px solid #00aa66;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #00ff88;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
}
.status-warn {
    background: rgba(255,140,0,0.1);
    border: 1px solid var(--orange);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: var(--orange);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
}
.status-err {
    background: rgba(255,51,102,0.1);
    border: 1px solid var(--red);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: var(--red);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
}

/* Divider */
.divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.footer {
    text-align: center;
    padding: 1rem;
    font-size: 0.75rem;
    color: #7f9bb3;
    font-family: 'Share Tech Mono', monospace;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.model.names = {0: 'Damaged', 1: 'Vial', 2: 'Sealed', 3: 'Unsealed'}
        return model, None
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
CLASS_COLORS = {
    'Vial':     (255, 220, 0),
    'Sealed':   (0, 255, 136),
    'Unsealed': (255, 140, 0),
    'Damaged':  (0, 0, 255),
}

def draw_boxes(img_bgr, results, model):
    """For image tab — Vial label above, quality label below"""
    names = model.model.names
    h, w = img_bgr.shape[:2]
    
    # Dynamic font scale based on image size
    font_scale = max(0.4, min(0.9, w / 1280))
    thickness = max(1, int(w / 800))
    box_thickness = max(2, int(w / 500))

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = names.get(cls, 'Unknown')
        color = CLASS_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, box_thickness)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if name == 'Vial':
            ly = max(y1 - 8, th + 8)
        else:
            ly = min(y2 + th + 10, h - 5)
        lx = min(x1, w - tw - 10)
        cv2.rectangle(img_bgr, (lx, ly - th - 4), (lx + tw + 8, ly + 2), color, -1)
        cv2.putText(img_bgr, label, (lx + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img_bgr

def draw_video_labels(frame, boxes, clss, confs, ids, class_map, class_colors, H):
    """
    For video tab — clear non-overlapping labels:
    Vial label → ABOVE the bounding box
    Quality label (Sealed/Unsealed/Damaged) → BELOW the bounding box
    """
    for box, cls, conf, tid in zip(boxes, clss, confs, ids):
        x1, y1, x2, y2 = map(int, box)
        name = class_map.get(cls, 'Vial')
        color = class_colors.get(name, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_txt = f"{name} {conf:.2f} | ID {tid}"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        if name == 'Vial':
            # Label ABOVE the bounding box
            ly = max(y1 - 5, th + 12)
            cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
            cv2.putText(frame, label_txt, (x1 + 4, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # Label BELOW the bounding box
            ly = min(y2 + th + 10, H - 5)
            cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
            cv2.putText(frame, label_txt, (x1 + 4, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def get_counts(results, model):
    counts = defaultdict(int)
    names = model.model.names
    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = names.get(cls, 'Unknown')
        counts[name] += 1
    return counts

def generate_html_report(vid_counts, video_name, fps, total_frames, W, H, conf_threshold, duration_sec):
    """Generate a professional HTML inspection report for download"""
    from datetime import datetime
    import random

    now        = datetime.now()
    date_str   = now.strftime("%d %B %Y")
    time_str   = now.strftime("%H:%M:%S")
    report_id  = f"VV-{now.strftime('%Y%m%d')}-{random.randint(1000,9999)}"

    total      = vid_counts.get('Vial', 0)
    sealed     = vid_counts.get('Sealed', 0)
    unsealed   = vid_counts.get('Unsealed', 0)
    damaged    = vid_counts.get('Damaged', 0)
    defective  = unsealed + damaged
    pass_rate  = (sealed / total * 100) if total > 0 else 0
    defect_rate= (defective / total * 100) if total > 0 else 0
    verdict    = "✅ PASS" if defect_rate < 5 else ("⚠️ WARNING" if defect_rate < 15 else "❌ FAIL")
    verdict_color = "#00ff88" if defect_rate < 5 else ("#ff8c00" if defect_rate < 15 else "#ff3366")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VialVision AI — Inspection Report {report_id}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:#f0f4f8; color:#1e293b; }}
  .page {{ max-width:900px; margin:0 auto; background:#fff; }}

  /* Header */
  .header {{ background:linear-gradient(135deg,#0d1b2a,#1b4a6e); color:#fff; padding:2.5rem 3rem; }}
  .header-top {{ display:flex; justify-content:space-between; align-items:flex-start; }}
  .logo {{ font-family:'Share Tech Mono',monospace; font-size:1.8rem; color:#00d4ff; letter-spacing:3px; }}
  .logo span {{ color:#00ff88; }}
  .report-id {{ font-family:'Share Tech Mono',monospace; font-size:0.85rem; color:#64a0c8; text-align:right; }}
  .report-id b {{ color:#00d4ff; font-size:1.1rem; display:block; }}
  .header-title {{ margin-top:1.2rem; font-size:1.3rem; font-weight:600; color:#e0f0ff; }}
  .header-meta {{ display:flex; gap:2rem; margin-top:0.6rem; font-size:0.85rem; color:#90b8d8; }}

  /* Verdict Banner */
  .verdict {{ background:#0f1e35; border-left:5px solid {verdict_color};
              padding:1rem 2rem; display:flex; justify-content:space-between; align-items:center; }}
  .verdict-label {{ font-size:0.8rem; color:#64a0c8; text-transform:uppercase; letter-spacing:2px; }}
  .verdict-value {{ font-family:'Share Tech Mono',monospace; font-size:1.6rem; color:{verdict_color}; font-weight:700; }}
  .verdict-rate {{ text-align:right; }}
  .verdict-rate .big {{ font-family:'Share Tech Mono',monospace; font-size:1.4rem; color:{verdict_color}; }}
  .verdict-rate .small {{ font-size:0.8rem; color:#64a0c8; }}

  /* Sections */
  .section {{ padding:2rem 3rem; border-bottom:1px solid #e2e8f0; }}
  .section-title {{ font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase;
                    letter-spacing:2px; margin-bottom:1.2rem; padding-bottom:0.5rem;
                    border-bottom:2px solid #e2e8f0; }}

  /* Stat Cards */
  .stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; }}
  .stat-card {{ border-radius:10px; padding:1.2rem; text-align:center; position:relative; overflow:hidden; }}
  .stat-card.total  {{ background:#eff6ff; border:1px solid #bfdbfe; }}
  .stat-card.sealed {{ background:#f0fdf4; border:1px solid #bbf7d0; }}
  .stat-card.unsealed{{ background:#fff7ed; border:1px solid #fed7aa; }}
  .stat-card.damaged {{ background:#fff1f2; border:1px solid #fecdd3; }}
  .stat-val {{ font-family:'Share Tech Mono',monospace; font-size:2.5rem; font-weight:700; line-height:1; }}
  .stat-card.total   .stat-val  {{ color:#1d4ed8; }}
  .stat-card.sealed  .stat-val  {{ color:#15803d; }}
  .stat-card.unsealed .stat-val {{ color:#c2410c; }}
  .stat-card.damaged  .stat-val {{ color:#be123c; }}
  .stat-label {{ font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:1.5px;
                 margin-top:0.4rem; color:#64748b; }}
  .stat-pct {{ font-size:0.85rem; font-weight:600; margin-top:0.3rem; }}
  .stat-card.sealed  .stat-pct {{ color:#15803d; }}
  .stat-card.unsealed .stat-pct {{ color:#c2410c; }}
  .stat-card.damaged  .stat-pct {{ color:#be123c; }}

  /* Progress bars */
  .bar-row {{ display:flex; align-items:center; gap:1rem; margin:0.6rem 0; }}
  .bar-label {{ width:90px; font-size:0.85rem; font-weight:600; color:#334155; }}
  .bar-track {{ flex:1; height:10px; background:#e2e8f0; border-radius:5px; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:5px; }}
  .bar-val {{ width:60px; text-align:right; font-family:'Share Tech Mono',monospace;
               font-size:0.85rem; color:#334155; }}

  /* Info Table */
  .info-table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
  .info-table tr {{ border-bottom:1px solid #f1f5f9; }}
  .info-table tr:last-child {{ border-bottom:none; }}
  .info-table td {{ padding:0.65rem 0; }}
  .info-table td:first-child {{ color:#64748b; width:220px; font-weight:500; }}
  .info-table td:last-child {{ color:#1e293b; font-family:'Share Tech Mono',monospace; font-size:0.85rem; }}

  /* Footer */
  .report-footer {{ background:#f8fafc; padding:1.5rem 3rem;
                    display:flex; justify-content:space-between; align-items:center; }}
  .report-footer .left {{ font-size:0.8rem; color:#64748b; line-height:1.8; }}
  .report-footer .right {{ font-size:0.75rem; color:#94a3b8; text-align:right; font-family:'Share Tech Mono',monospace; }}

  @media print {{
    body {{ background:#fff; }}
    .page {{ box-shadow:none; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <div class="header-top">
      <div>
        <div class="logo">💊 VialVision<span>AI</span></div>
        <div style="font-size:0.8rem;color:#64a0c8;margin-top:0.3rem;letter-spacing:1px;">
          PHARMACEUTICAL INSPECTION SYSTEM
        </div>
      </div>
      <div class="report-id">
        REPORT ID<b>{report_id}</b>
        {date_str}<br>{time_str}
      </div>
    </div>
    <div class="header-title">Quality Control Inspection Report</div>
    <div class="header-meta">
      <span>📹 {video_name}</span>
      <span>🏭 Ramaiah Institute of Technology</span>
      <span>🤖 Model: YOLOv11s</span>
    </div>
  </div>

  <!-- VERDICT BANNER -->
  <div class="verdict">
    <div>
      <div class="verdict-label">Overall Quality Verdict</div>
      <div class="verdict-value">{verdict}</div>
    </div>
    <div style="text-align:center;">
      <div class="verdict-label">Pass Rate</div>
      <div class="verdict-value">{pass_rate:.1f}%</div>
    </div>
    <div class="verdict-rate">
      <div class="verdict-label">Defect Rate</div>
      <div class="big">{defect_rate:.1f}%</div>
      <div class="small">Threshold: &lt;5% = PASS</div>
    </div>
  </div>

  <!-- DETECTION SUMMARY -->
  <div class="section">
    <div class="section-title">📊 Detection Summary</div>
    <div class="stat-grid">
      <div class="stat-card total">
        <div class="stat-val">{total}</div>
        <div class="stat-label">Total Vials</div>
        <div class="stat-pct" style="color:#1d4ed8;">Inspected</div>
      </div>
      <div class="stat-card sealed">
        <div class="stat-val">{sealed}</div>
        <div class="stat-label">Sealed</div>
        <div class="stat-pct">{sealed/total*100:.1f}% ✓</div>
      </div>
      <div class="stat-card unsealed">
        <div class="stat-val">{unsealed}</div>
        <div class="stat-label">Unsealed</div>
        <div class="stat-pct">{unsealed/total*100:.1f}%</div>
      </div>
      <div class="stat-card damaged">
        <div class="stat-val">{damaged}</div>
        <div class="stat-label">Damaged</div>
        <div class="stat-pct">{damaged/total*100:.1f}%</div>
      </div>
    </div>

    <!-- Distribution bars -->
    <div style="margin-top:1.5rem;">
      <div class="bar-row">
        <div class="bar-label">Sealed</div>
        <div class="bar-track"><div class="bar-fill" style="width:{sealed/total*100:.1f}%;background:#22c55e;"></div></div>
        <div class="bar-val">{sealed/total*100:.1f}%</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Unsealed</div>
        <div class="bar-track"><div class="bar-fill" style="width:{unsealed/total*100:.1f}%;background:#f97316;"></div></div>
        <div class="bar-val">{unsealed/total*100:.1f}%</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Damaged</div>
        <div class="bar-track"><div class="bar-fill" style="width:{damaged/total*100:.1f}%;background:#f43f5e;"></div></div>
        <div class="bar-val">{damaged/total*100:.1f}%</div>
      </div>
    </div>
  </div>

  <!-- VIDEO & MODEL INFO -->
  <div class="section">
    <div class="section-title">🎥 Inspection Details</div>
    <table class="info-table">
      <tr><td>Inspection Date</td><td>{date_str}</td></tr>
      <tr><td>Inspection Time</td><td>{time_str}</td></tr>
      <tr><td>Video File</td><td>{video_name}</td></tr>
      <tr><td>Video Resolution</td><td>{W} × {H} px</td></tr>
      <tr><td>Frame Rate</td><td>{fps} FPS</td></tr>
      <tr><td>Total Frames</td><td>{total_frames}</td></tr>
      <tr><td>Video Duration</td><td>{duration_sec:.1f} seconds ({duration_sec/60:.2f} min)</td></tr>
      <tr><td>Confidence Threshold</td><td>{conf_threshold}</td></tr>
      <tr><td>AI Model</td><td>YOLOv11s (Ultralytics)</td></tr>
      <tr><td>Model Performance</td><td>mAP@50: 96.9% | mAP@50-95: 94.1%</td></tr>
      <tr><td>Inference Speed</td><td>~106 FPS on GPU</td></tr>
      <tr><td>Tracking Algorithm</td><td>BotSort (Multi-Object Tracker)</td></tr>
      <tr><td>Counting Method</td><td>Virtual Line Crossing</td></tr>
    </table>
  </div>

  <!-- QUALITY STANDARDS -->
  <div class="section">
    <div class="section-title">📋 Quality Standards Reference</div>
    <table class="info-table">
      <tr><td>Defect Rate &lt; 5%</td><td style="color:#15803d;font-weight:600;">✅ PASS — Acceptable Quality</td></tr>
      <tr><td>Defect Rate 5–15%</td><td style="color:#c2410c;font-weight:600;">⚠️ WARNING — Review Required</td></tr>
      <tr><td>Defect Rate &gt; 15%</td><td style="color:#be123c;font-weight:600;">❌ FAIL — Production Hold</td></tr>
      <tr><td>Compliance</td><td>FDA GMP | WHO Guidelines | ISO 15378</td></tr>
      <tr><td>This Batch Result</td><td style="color:{verdict_color};font-weight:700;">{verdict} — Defect Rate: {defect_rate:.1f}%</td></tr>
    </table>
  </div>

  <!-- FOOTER -->
  <div class="report-footer">
    <div class="left">
      <b>VialVision AI</b> — Pharmaceutical Quality Control System<br>
      Ramaiah Institute of Technology, Bengaluru<br>
      M.Tech Robotics & AI | Developed by Geeta Siddramayya Math
    </div>
    <div class="right">
      Report ID: {report_id}<br>
      Generated: {date_str} {time_str}<br>
      YOLOv11s | 96.9% mAP50
    </div>
  </div>

</div>
</body>
</html>"""
    return html


def generate_csv_report(vid_counts, video_name, fps, total_frames, W, H, conf_threshold, duration_sec):
    """Generate CSV report for spreadsheet/database logging"""
    from datetime import datetime
    import io, csv, random

    now       = datetime.now()
    report_id = f"VV-{now.strftime('%Y%m%d')}-{random.randint(1000,9999)}"
    total     = vid_counts.get('Vial', 0)
    sealed    = vid_counts.get('Sealed', 0)
    unsealed  = vid_counts.get('Unsealed', 0)
    damaged   = vid_counts.get('Damaged', 0)
    defective = unsealed + damaged
    pass_rate = (sealed / total * 100) if total > 0 else 0
    defect_rate = (defective / total * 100) if total > 0 else 0
    verdict   = "PASS" if defect_rate < 5 else ("WARNING" if defect_rate < 15 else "FAIL")

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["VIALVISION AI — INSPECTION REPORT"])
    writer.writerow([])
    writer.writerow(["Report ID",         report_id])
    writer.writerow(["Date",              now.strftime("%d/%m/%Y")])
    writer.writerow(["Time",              now.strftime("%H:%M:%S")])
    writer.writerow(["Video File",        video_name])
    writer.writerow([])
    writer.writerow(["=== DETECTION RESULTS ==="])
    writer.writerow(["Metric",            "Count",    "Percentage"])
    writer.writerow(["Total Vials",       total,      "100%"])
    writer.writerow(["Sealed",            sealed,     f"{pass_rate:.1f}%"])
    writer.writerow(["Unsealed",          unsealed,   f"{unsealed/total*100:.1f}%" if total>0 else "0%"])
    writer.writerow(["Damaged",           damaged,    f"{damaged/total*100:.1f}%"  if total>0 else "0%"])
    writer.writerow(["Total Defective",   defective,  f"{defect_rate:.1f}%"])
    writer.writerow([])
    writer.writerow(["=== QUALITY VERDICT ==="])
    writer.writerow(["Pass Rate",         f"{pass_rate:.1f}%"])
    writer.writerow(["Defect Rate",       f"{defect_rate:.1f}%"])
    writer.writerow(["Verdict",           verdict])
    writer.writerow([])
    writer.writerow(["=== VIDEO INFO ==="])
    writer.writerow(["Resolution",        f"{W}x{H}"])
    writer.writerow(["FPS",               fps])
    writer.writerow(["Total Frames",      total_frames])
    writer.writerow(["Duration (sec)",    f"{duration_sec:.1f}"])
    writer.writerow(["Confidence",        conf_threshold])
    writer.writerow(["Model",             "YOLOv11s"])
    writer.writerow(["mAP50",             "96.9%"])

    return output.getvalue()


def conf_bar_html(name, conf, color):
    pct = int(conf * 100)
    return f"""
    <div class="conf-bar-wrap">
        <div class="conf-label"><span>{name}</span><span>{pct}%</span></div>
        <div class="conf-bar"><div class="conf-fill" style="width:{pct}%;background:{color}"></div></div>
    </div>"""

def metric_cards_html(counts):
    total_vials = counts.get('Vial', 0)
    sealed      = counts.get('Sealed', 0)
    unsealed    = counts.get('Unsealed', 0)
    Damaged     = counts.get('Damaged', 0)
    return f"""
    <div class="metric-grid">
        <div class="metric-card vial">
            <div class="metric-value">{total_vials}</div>
            <div class="metric-label">Vials</div>
        </div>
        <div class="metric-card sealed">
            <div class="metric-value">{sealed}</div>
            <div class="metric-label">Sealed</div>
        </div>
        <div class="metric-card unsealed">
            <div class="metric-value">{unsealed}</div>
            <div class="metric-label">Unsealed</div>
        </div>
        <div class="metric-card Damaged">
            <div class="metric-value">{Damaged}</div>
            <div class="metric-label">Damaged</div>
        </div>
    </div>"""

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div class="header-title">💊 VialVision AI</div>
    <div class="header-sub">AI Powered Pharmaceutical Vial Detection, Counting & Quality Control System</div>
    <span class="header-badge">YOLOv11s</span>
    <span class="header-badge">96.9% mAP50</span>
    <span class="header-badge">Real-Time</span>
    <span class="header-badge">Ramaiah Institute of Technology</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="panel-title">⚙ Configuration</div>', unsafe_allow_html=True)

    model_path = st.text_input(
        "Model Path",
        value="best.pt",
        help="Path to your trained best.pt file"
    )

    conf_threshold = 0.5
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📊 Model Info</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.8rem; line-height:2;">
    <span style="color:#00d4ff">Model:</span> YOLOv11s<br>
    <span style="color:#00d4ff">Classes:</span> 4<br>
    <span style="color:#00ff88">● Sealed</span><br>
    <span style="color:#ff8c00">● Unsealed</span><br>
    <span style="color:#ff3366">● Damaged</span><br>
    <span style="color:#00d4ff">● Vial</span><br>
    <span style="color:#00d4ff">mAP50:</span> 96.9%<br>
    <span style="color:#00d4ff">Speed:</span> ~106 FPS
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📋 Class Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem;">
    <b style="color:#00d4ff">0</b> → Damaged<br>
    <b style="color:#00d4ff">1</b> → Vial<br>
    <b style="color:#00d4ff">2</b> → Sealed<br>
    <b style="color:#00d4ff">3</b> → Unsealed
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, err = load_model(model_path)

if err:
    st.markdown(f'<div class="status-err">⚠ Model Error: {err}<br>Please check the model path in sidebar.</div>', unsafe_allow_html=True)
    st.stop()
else:
    st.markdown('<div class="status-ok">✓ Model loaded successfully — Ready for inference</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸  IMAGE DETECTION", "🎥  VIDEO DETECTION", "📷  WEBCAM DETECTION"])

# ══════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<br>', unsafe_allow_html=True)
    uploaded_imgs = st.file_uploader(
        "Upload vial images",
        type=['jpg', 'jpeg', 'png', 'bmp', 'jfif', 'webp'],
        accept_multiple_files=True,
        key="img_upload"
    )

    if uploaded_imgs:
        for uploaded_file in uploaded_imgs:
            st.markdown(f'<div class="panel-title">🔍 {uploaded_file.name}</div>', unsafe_allow_html=True)

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img_bgr is None:
                uploaded_file.seek(0)
                pil_img = Image.open(uploaded_file).convert('RGB')
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            with st.spinner("Running inference..."):
                results = model.predict(img_bgr, conf=conf_threshold, verbose=False)

            img_out = img_bgr.copy()
            img_out = draw_boxes(img_out, results, model)
            img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(img_rgb, caption="Detection Result", use_container_width=True)

            with col2:
                counts = get_counts(results, model)
                st.markdown(metric_cards_html(counts), unsafe_allow_html=True)

                st.markdown('<div class="panel-title" style="margin-top:1rem">DETECTIONS</div>', unsafe_allow_html=True)
                if len(results[0].boxes) == 0:
                    st.markdown('<div class="status-warn">No objects detected</div>', unsafe_allow_html=True)
                else:
                    names = model.model.names
                    conf_html = ""
                    for box in results[0].boxes:
                        cls  = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = names.get(cls, 'Unknown')
                        color_map = {'Vial': '#00d4ff', 'Sealed': '#00ff88', 'Unsealed': '#ff8c00', 'Damaged': '#ff3366'}
                        color = color_map.get(name, '#ffffff')
                        conf_html += conf_bar_html(name, conf, color)
                    st.markdown(conf_html, unsafe_allow_html=True)

                    tags_html = ""
                    for name, cnt in counts.items():
                        cls_name = name.lower()
                        tags_html += f'<span class="det-tag {cls_name}">{name} × {cnt}</span>'
                    st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — VIDEO  (updated tracking + clear labels)
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<br>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader(
        "Upload a vial inspection video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="vid_upload"
    )

    if uploaded_video:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Force 720p for dashboard clarity
        W, H = 1280, 720

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">Video Info</div>
            <div style="font-size:0.85rem; font-family:'Share Tech Mono',monospace; line-height:2; color:#c8d8e8;">
            Resolution: <span style="color:#00d4ff">{W}×{H}</span> &nbsp;|&nbsp;
            FPS: <span style="color:#00d4ff">{fps}</span> &nbsp;|&nbsp;
            Frames: <span style="color:#00d4ff">{total_frames}</span> &nbsp;|&nbsp;
            Duration: <span style="color:#00d4ff">{total_frames/fps:.1f}s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 1])
        with col_l:
            enable_counting = st.checkbox("Enable Vial Counting (Virtual Line)", value=True)
            line_position   = st.slider("Counting Line Position (%)", 20, 80, 50) if enable_counting else 50
        with col_r:
            process_every = st.selectbox("Process every N frames", [1, 2, 3, 5], index=0)

        if st.button("▶  Process Video", use_container_width=True):

            # ── Class config ──────────────────────────────────────
            CLASS_MAP = {0: 'Damaged', 1: 'Vial', 2: 'Sealed', 3: 'Unsealed'}

            # ── Tracking state ───────────────────────────────────
            line_x       = int(line_position / 100 * W)
            last_x       = {}
            counted_ids  = set()
            vid_counts   = defaultdict(int)

            # ── Output video writer ───────────────────────────────
            out_path = tempfile.mktemp(suffix='_output.mp4')
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

            cap              = cv2.VideoCapture(tfile.name)
            progress_bar     = st.progress(0)
            status_text      = st.empty()
            preview_placeholder = st.empty()

            frame_count    = 0
            cached_results = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Resize to fixed 720p (INTER_CUBIC preserves defect edges)
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_CUBIC)

                progress = frame_count / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.markdown(
                    f'<div style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#00d4ff">'
                    f'Processing frame {frame_count}/{total_frames} — {int(progress*100)}%</div>',
                    unsafe_allow_html=True
                )

                # Run BotSort tracking every N frames
                if frame_count % process_every == 0:
                    cached_results = model.track(
                        frame, persist=True,
                        tracker="botsort.yaml",
                        conf=conf_threshold,
                        verbose=False
                    )

                if cached_results is not None:

                    # Draw counting line
                    if enable_counting:
                        cv2.line(frame, (line_x, 0), (line_x, H), (0, 0, 255), 2)

                    if (cached_results[0].boxes is not None and
                            cached_results[0].boxes.id is not None):

                        boxes = cached_results[0].boxes.xyxy.cpu().numpy()
                        clss  = cached_results[0].boxes.cls.cpu().numpy().astype(int)
                        confs = cached_results[0].boxes.conf.cpu().numpy()
                        ids   = cached_results[0].boxes.id.cpu().numpy().astype(int)

                        # ── Gather quality labels in this frame ──────────
                        frame_defects = []
                        for b, c in zip(boxes, clss):
                            nm = CLASS_MAP.get(c)
                            if nm in ['Sealed', 'Unsealed', 'Damaged']:
                                frame_defects.append((b, nm))

                        # ── Counting logic (virtual line crossing) ────────
                        for box, cls, conf, tid in zip(boxes, clss, confs, ids):
                            x1, y1, x2, y2 = map(int, box)
                            cx = (x1 + x2) // 2
                            name = CLASS_MAP.get(cls, 'Vial')

                            if enable_counting and name == 'Vial':
                                if tid in last_x and tid not in counted_ids:
                                    prev_cx = last_x[tid]
                                    crossed = (
                                        (prev_cx >= line_x and cx < line_x) or
                                        (prev_cx <= line_x and cx > line_x)
                                    )
                                    if crossed:
                                        # Match nearest quality label to this vial
                                        best_state = 'Sealed'
                                        min_dist   = 100
                                        for d_box, d_name in frame_defects:
                                            d_cx = (d_box[0] + d_box[2]) // 2
                                            dist = abs(cx - d_cx)
                                            if dist < min_dist:
                                                best_state = d_name
                                                min_dist   = dist
                                        vid_counts[best_state] += 1
                                        counted_ids.add(tid)
                                last_x[tid] = cx

                        # ── Draw labels: Vial ABOVE, Quality BELOW ────────
                        frame = draw_video_labels(
                            frame, boxes, clss, confs, ids,
                            CLASS_MAP, CLASS_COLORS, H
                        )

                    # ── Dashboard overlay (top-right corner) ─────────────
                    if enable_counting:
                        total   = sum(vid_counts.values())
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (W - 220, 10), (W - 10, 140), (0, 0, 0), -1)
                        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
                        f_style = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, f"Total:    {total}",
                                    (W - 210, 35),  f_style, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Sealed:   {vid_counts['Sealed']}",
                                    (W - 210, 62),  f_style, 0.5, (0, 255, 136), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Unsealed: {vid_counts['Unsealed']}",
                                    (W - 210, 89),  f_style, 0.5, (255, 140, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Damaged:  {vid_counts['Damaged']}",
                                    (W - 210, 116), f_style, 0.5, (0, 0, 255),   1, cv2.LINE_AA)

                out.write(frame)

                # Live preview every 30 frames
                if frame_count % 30 == 0:
                    preview_placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_count} | Counted: {sum(vid_counts.values())}",
                        use_container_width=True
                    )

            cap.release()
            out.release()
            progress_bar.progress(1.0)
            status_text.empty()

            # Final counts — Vial = total of all quality states
            vid_counts['Vial'] = vid_counts['Sealed'] + vid_counts['Unsealed'] + vid_counts['Damaged']

            duration_sec = total_frames / fps if fps > 0 else 0

            # ── Save everything to session_state so buttons persist after rerun ──
            with open(out_path, 'rb') as f:
                st.session_state['video_bytes'] = f.read()

            st.session_state['html_report'] = generate_html_report(
                vid_counts, uploaded_video.name,
                fps, total_frames, W, H, conf_threshold, duration_sec
            )
            st.session_state['csv_report'] = generate_csv_report(
                vid_counts, uploaded_video.name,
                fps, total_frames, W, H, conf_threshold, duration_sec
            )
            st.session_state['vid_counts']    = dict(vid_counts)
            st.session_state['processing_done'] = True

        # ── Download buttons — OUTSIDE if block so they survive reruns ──────
        if st.session_state.get('processing_done', False):
            st.markdown('<div class="status-ok">✓ Video processing complete — Download your files below</div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)

            # Metric cards
            st.markdown(metric_cards_html(st.session_state['vid_counts']), unsafe_allow_html=True)

            dl_col1, dl_col2, dl_col3 = st.columns(3)

            with dl_col1:
                st.download_button(
                    "⬇  Download Processed Video",
                    st.session_state['video_bytes'],
                    file_name="vial_detection_output.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key="dl_video"
                )

            with dl_col2:
                st.download_button(
                    "📄  Download HTML Report",
                    st.session_state['html_report'].encode('utf-8'),
                    file_name="vialvision_inspection_report.html",
                    mime="text/html",
                    use_container_width=True,
                    key="dl_html"
                )

            with dl_col3:
                st.download_button(
                    "📊  Download CSV Report",
                    st.session_state['csv_report'].encode('utf-8'),
                    file_name="vialvision_inspection_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_csv"
                )

            # ── Report preview ────────────────────────────────────
            with st.expander("👁  Preview Inspection Report"):
                import streamlit.components.v1 as components
                components.html(st.session_state['html_report'], height=900, scrolling=True)

# ══════════════════════════════════════════════
# TAB 3 — LIVE CAMERA (Logitech)
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
        <div class="panel-title">📷 Live Camera Detection</div>
        <div style="font-size:0.85rem;color:#4a6080;line-height:1.8;">
        <span style="color:#00d4ff">• Connect your Logitech camera before starting</span><br>
        <span style="color:#00ff88">• Camera index 0 = Logitech external, 1 = built-in</span><br>
        <span style="color:#ffe066">• Uses YOLOv11s detection on each frame</span><br>
        <span style="color:#00d4ff">• Press <b style="color:#ff3366">Stop Camera</b> to end session
        </div>
    </div>
    """, unsafe_allow_html=True)

    cam_index = st.selectbox("Camera Index", [0, 1, 2],
                              index=1, help="0=Logitech external, 1=built-in")

    col1, col2 = st.columns(2)
    start_cam = col1.button("▶  Start Camera", use_container_width=True)
    stop_cam  = col2.button("⏹  Stop Camera",  use_container_width=True)

    if start_cam:
        st.session_state['cam_running'] = True
    if stop_cam:
        st.session_state['cam_running'] = False

    if st.session_state.get('cam_running', False):
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.markdown(
                '<div class="status-err">⚠ Cannot access camera. Try index 0 or 1.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-ok">✓ Camera active — Live detection running</div>',
                unsafe_allow_html=True
            )
            cam_ph     = st.empty()
            metrics_ph = st.empty()
            frame_cnt  = 0

            while st.session_state.get('cam_running', False):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_cnt += 1

                # YOUR EXACT: Run detection every 2 frames
                if frame_cnt % 2 == 0:
                    results = model.predict(frame, verbose=False, conf=conf_threshold)

                    # Draw labels using draw_boxes
                    if len(results[0].boxes) > 0:
                        frame = draw_boxes(frame.copy(), results, model)

                    # YOUR EXACT: Count by class
                    class_counts = {}
                    for box in results[0].boxes:
                        cls = model.model.names[int(box.cls[0])]
                        class_counts[cls] = class_counts.get(cls, 0) + 1

                    cam_ph.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        caption="Live Detection",
                        use_container_width=True
                    )
                    metrics_ph.markdown(
                        metric_cards_html(class_counts),
                        unsafe_allow_html=True
                    )
            cap.release()
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;border:1px dashed #1e3a5f;border-radius:10px;">
            <div style="font-size:3rem;">📷</div>
            <div style="color:#c8d8e8;font-family:'Share Tech Mono',monospace;
                        font-size:0.85rem;margin-top:1rem;">
                CAMERA INACTIVE<br>Click "Start Camera" to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
© 2026 VialVision AI &nbsp;|&nbsp; Developed by Geeta Math &nbsp;|&nbsp;
Ramaiah Institute of Technology &nbsp;|&nbsp; Robotics and AI
</div>
""", unsafe_allow_html=True)
