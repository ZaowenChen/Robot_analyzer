"""
Report Generator — HTML Diagnostic Reports with Interactive Timeline

Consumes the output from cross_validator.py and/or analyze.py and produces
a self-contained HTML report with:
  - Executive summary dashboard (verdict counts, severity breakdown)
  - Interactive timeline (SVG-based, zoomable)
  - Incident/evidence cards with expandable details
  - Foxglove layout download links (when paired with layout generator)

Usage:
    from reporting.html_report import generate_html_report

    # From cross_validator output
    generate_html_report(
        cross_validation_report="cross_validation_report.json",
        output_path="report.html",
    )

    # From analyze.py output
    generate_html_report(
        diagnostic_report="diagnostic_report.json",
        output_path="report.html",
    )
"""

import json
import os
import html as html_lib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from core.utils import CST, format_absolute_time


# ---------------------------------------------------------------------------
# Color / severity helpers
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "CRITICAL": "#dc2626",
    "WARNING": "#f59e0b",
    "INFO": "#3b82f6",
}

VERDICT_COLORS = {
    "CONFIRMED": "#16a34a",
    "CONTRADICTED": "#dc2626",
    "NO_SENSOR_DATA": "#9ca3af",
    "UNCHECKED": "#d1d5db",
}

CATEGORY_ICONS = {
    "HW_FAULT": "wrench",
    "HARDWARE_FAULT": "wrench",
    "IMU": "compass",
    "IMU_FROZEN": "compass",
    "IMU_HW_FIELD_ZERO": "compass",
    "NAVIGATION": "map",
    "NAVIGATION_STUCK": "map",
    "NAV_STUCK": "map",
    "LOCALIZATION": "crosshair",
    "LOCALIZATION_STUCK": "crosshair",
    "SENSOR_FREEZE": "thermometer",
    "FROZEN_SENSOR": "thermometer",
    "FREEZE_ONSET": "thermometer",
    "SENSOR_RESUME": "activity",
    "SAFETY": "shield",
    "SAFETY_STATE": "shield",
    "LIDAR": "radio",
    "ZERO_FIELD": "minus-circle",
    "UNSTABLE_FREQUENCY": "trending-down",
    "IR_SENSOR_DEAD": "wifi-off",
    "MOTION": "move",
    "MOTION_STATE": "move",
    "ERROR": "alert-triangle",
    "FAILURE": "alert-triangle",
    "INFO": "info",
    "HIGH_FREQUENCY_EVENT": "repeat",
    "BATTERY_STATE": "battery",
    "OBSTACLE_DETECT": "alert-octagon",
    "DEPTH_CAMERA": "radio",
    "DEPTH_CAMERA_ERROR": "radio",
    "DEPTHCAM_FUSION_FAIL": "radio",
    "DL_INFER_EVENT": "radio",
    "LOCALIZATION_STATUS": "crosshair",
    "MAPPING_STATE": "map",
}


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html_lib.escape(str(text))


def _severity_badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#6b7280")
    return f'<span class="badge" style="background:{color}">{_esc(severity)}</span>'


def _verdict_badge(verdict: str) -> str:
    color = VERDICT_COLORS.get(verdict, "#6b7280")
    return f'<span class="badge" style="background:{color}">{_esc(verdict)}</span>'


def _icon_svg(category: str) -> str:
    """Return a small inline SVG icon based on the category. Uses Feather-style icons."""
    # We use simple unicode + CSS instead of actual SVGs for self-containment
    icon_map = {
        "wrench": "\U0001f527",
        "compass": "\U0001f9ed",
        "map": "\U0001f5fa\ufe0f",
        "crosshair": "\U0001f3af",
        "thermometer": "\U0001f321\ufe0f",
        "activity": "\U0001f4c8",
        "shield": "\U0001f6e1\ufe0f",
        "radio": "\U0001f4e1",
        "minus-circle": "\u2296",
        "trending-down": "\U0001f4c9",
        "wifi-off": "\U0001f4f4",
        "move": "\u2194\ufe0f",
        "alert-triangle": "\u26a0\ufe0f",
        "info": "\u2139\ufe0f",
        "repeat": "\U0001f501",
        "battery": "\U0001f50b",
        "alert-octagon": "\U0001f6d1",
    }
    icon_name = CATEGORY_ICONS.get(category, "info")
    emoji = icon_map.get(icon_name, "\u2139\ufe0f")
    return f'<span class="cat-icon">{emoji}</span>'


# ---------------------------------------------------------------------------
# Timeline SVG generation
# ---------------------------------------------------------------------------

def _build_timeline_svg(events: List[dict], width: int = 1200, height: int = 180) -> str:
    """Build an SVG timeline from timestamped events."""
    if not events:
        return '<p class="muted">No timeline events to display.</p>'

    timestamps = [e.get("timestamp", 0) for e in events if isinstance(e.get("timestamp"), (int, float)) and e["timestamp"] > 0]
    if not timestamps:
        return '<p class="muted">No valid timestamps in events.</p>'

    t_min = min(timestamps)
    t_max = max(timestamps)
    span = max(t_max - t_min, 1.0)

    margin_left = 20
    margin_right = 20
    usable = width - margin_left - margin_right
    lane_y = {"CRITICAL": 40, "WARNING": 80, "INFO": 120}

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="timeline-svg" '
        f'xmlns="http://www.w3.org/2000/svg">',
        # Background
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#1e1e2e" rx="8"/>',
        # Lane labels
        f'<text x="10" y="38" fill="#dc2626" font-size="10" font-family="monospace">CRIT</text>',
        f'<text x="10" y="78" fill="#f59e0b" font-size="10" font-family="monospace">WARN</text>',
        f'<text x="10" y="118" fill="#3b82f6" font-size="10" font-family="monospace">INFO</text>',
        # Lane lines
        f'<line x1="{margin_left}" y1="45" x2="{width - margin_right}" y2="45" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="85" x2="{width - margin_right}" y2="85" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="125" x2="{width - margin_right}" y2="125" stroke="#333" stroke-width="1"/>',
    ]

    # Time axis labels
    num_ticks = min(8, max(2, int(span / 60)))
    for i in range(num_ticks + 1):
        t = t_min + (span * i / num_ticks)
        x = margin_left + (t - t_min) / span * usable
        dt = datetime.fromtimestamp(t, tz=CST)
        label = dt.strftime("%H:%M:%S")
        parts.append(f'<line x1="{x:.1f}" y1="135" x2="{x:.1f}" y2="145" stroke="#555" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="160" fill="#888" font-size="9" '
                     f'font-family="monospace" text-anchor="middle">{label}</text>')

    # Event dots
    for ev in events:
        ts = ev.get("timestamp", 0)
        if not isinstance(ts, (int, float)) or ts <= 0:
            continue
        sev = ev.get("severity", "INFO")
        x = margin_left + (ts - t_min) / span * usable
        y = lane_y.get(sev, 120)
        color = SEVERITY_COLORS.get(sev, "#3b82f6")
        cat = _esc(ev.get("category", ""))
        desc_short = _esc(str(ev.get("description", ev.get("summary", "")))[:60])
        ts_str = _esc(ev.get("timestamp_str", ""))
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y}" r="4" fill="{color}" '
            f'opacity="0.85" class="evt-dot">'
            f'<title>[{ts_str}] {cat}: {desc_short}</title>'
            f'</circle>'
        )

    parts.append('</svg>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Card builders
# ---------------------------------------------------------------------------

def _evidence_packet_card(packet: dict, index: int) -> str:
    """Render an EvidencePacket as an expandable HTML card."""
    pid = _esc(packet.get("packet_id", f"EP-{index:04d}"))
    sev = packet.get("severity", "INFO")
    verdict = packet.get("verdict", "UNCHECKED")
    cat = packet.get("category", "")
    summary = _esc(packet.get("summary", ""))
    ts = _esc(packet.get("timestamp", ""))
    bag = _esc(packet.get("bag_name", ""))
    dedup = packet.get("log_dedup_count", 1)

    log_ev = packet.get("log_event", {})
    msg = _esc(str(log_ev.get("message", ""))[:300])
    node = _esc(log_ev.get("node", ""))
    level = _esc(log_ev.get("level", ""))

    # Cross-validation details
    cv_details = packet.get("cross_validation_details", [])
    cv_html = ""
    if cv_details:
        cv_rows = []
        for d in cv_details:
            v = d.get("verdict", "")
            cv_rows.append(
                f'<tr><td>{_esc(d.get("rule", ""))}</td>'
                f'<td>{_verdict_badge(v)}</td>'
                f'<td>{_esc(d.get("detail", ""))}</td></tr>'
            )
        cv_html = f'''
        <div class="cv-details">
            <h4>Cross-Validation Rules</h4>
            <table class="cv-table">
                <tr><th>Rule</th><th>Verdict</th><th>Detail</th></tr>
                {"".join(cv_rows)}
            </table>
        </div>'''

    # Robot state
    robot_state = packet.get("robot_state", {})
    state_html = ""
    if robot_state:
        state_items = ", ".join(f"<code>{_esc(k)}={_esc(v)}</code>" for k, v in robot_state.items())
        state_html = f'<div class="state-ctx">Robot State: {state_items}</div>'

    dedup_html = f' <span class="dedup-badge">x{dedup}</span>' if dedup > 1 else ""

    return f'''
    <details class="evidence-card sev-{sev.lower()}" id="{pid}">
        <summary>
            {_icon_svg(cat)}
            <span class="ep-id">{pid}</span>
            {_severity_badge(sev)}
            {_verdict_badge(verdict)}
            <span class="ep-cat">{_esc(cat)}</span>
            <span class="ep-ts">{ts}</span>
            <span class="ep-bag">{bag}</span>
            {dedup_html}
        </summary>
        <div class="card-body">
            <div class="log-msg">
                <strong>[{level}] [{node}]</strong> {msg}
            </div>
            {state_html}
            {cv_html}
            <pre class="summary-pre">{summary}</pre>
        </div>
    </details>'''


def _incident_card(incident: dict, index: int) -> str:
    """Render an Incident (from analyze.py) as an expandable HTML card."""
    iid = _esc(incident.get("incident_id", f"INC-{index:03d}"))
    title = _esc(incident.get("title", "Unknown Incident"))
    sev = incident.get("severity", "INFO")
    cat = incident.get("category", "")
    root_cause = _esc(incident.get("root_cause", ""))
    time_start = _esc(incident.get("time_start", ""))
    time_end = _esc(incident.get("time_end", ""))
    duration = incident.get("duration_sec", 0)
    bags = incident.get("bags", [])
    actions = incident.get("recommended_actions", [])
    suppressed = incident.get("suppressed", False)
    suppression_reason = _esc(incident.get("suppression_reason", ""))
    raw_count = incident.get("raw_anomaly_count", 0)
    is_cross_bag = incident.get("is_cross_bag", False)

    actions_html = ""
    if actions:
        items = "".join(f"<li>{_esc(a)}</li>" for a in actions)
        actions_html = f'<div class="actions"><h4>Recommended Actions</h4><ul>{items}</ul></div>'

    suppressed_html = ""
    if suppressed:
        suppressed_html = f'<div class="suppressed-notice">Suppressed: {suppression_reason}</div>'

    cross_badge = ' <span class="badge" style="background:#8b5cf6">CROSS-BAG</span>' if is_cross_bag else ""

    log_evidence = incident.get("log_evidence", [])
    sensor_evidence = incident.get("sensor_evidence", [])

    evidence_html = ""
    if log_evidence or sensor_evidence:
        evidence_rows = []
        for le in log_evidence[:10]:
            evidence_rows.append(f'<div class="evidence-item log-ev">[LOG] {_esc(str(le)[:200])}</div>')
        for se in sensor_evidence[:10]:
            evidence_rows.append(f'<div class="evidence-item sensor-ev">[SENSOR] {_esc(str(se)[:200])}</div>')
        evidence_html = f'<div class="evidence-list">{"".join(evidence_rows)}</div>'

    return f'''
    <details class="incident-card sev-{sev.lower()}" id="{iid}" {"open" if sev == "CRITICAL" else ""}>
        <summary>
            {_icon_svg(cat)}
            <span class="ep-id">{iid}</span>
            {_severity_badge(sev)}
            {cross_badge}
            <strong>{title}</strong>
            <span class="ep-ts">{time_start}</span>
        </summary>
        <div class="card-body">
            <div class="incident-meta">
                <span>Duration: <strong>{duration:.1f}s</strong></span>
                <span>Raw anomalies: <strong>{raw_count}</strong></span>
                <span>Bags: <code>{_esc(", ".join(bags))}</code></span>
            </div>
            <div class="root-cause"><strong>Root Cause:</strong> {root_cause}</div>
            {suppressed_html}
            {actions_html}
            {evidence_html}
        </div>
    </details>'''


def _anomaly_card(anomaly: dict) -> str:
    """Render a timeline anomaly (from mission-mode diagnostic_report.json) as a card."""
    sev = anomaly.get("severity", "INFO")
    cat = anomaly.get("category", "")
    desc = _esc(anomaly.get("description", ""))
    ts = _esc(anomaly.get("timestamp_str", ""))
    bag = _esc(anomaly.get("bag", ""))
    details = anomaly.get("details", {})

    detail_items = ""
    if details:
        items = "".join(f"<span class='detail-chip'>{_esc(k)}: {_esc(str(v))}</span>" for k, v in details.items())
        detail_items = f'<div class="detail-chips">{items}</div>'

    return f'''
    <div class="anomaly-row sev-{sev.lower()}">
        {_icon_svg(cat)}
        {_severity_badge(sev)}
        <span class="anom-ts">{ts}</span>
        <span class="anom-cat">{_esc(cat)}</span>
        <span class="anom-desc">{desc}</span>
        <span class="anom-bag">{bag}</span>
        {detail_items}
    </div>'''


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Robot Diagnostic Report</title>
<style>
:root {{
    --bg: #0f0f17;
    --card-bg: #1a1a2e;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --border: #2d2d44;
    --accent: #3b82f6;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 24px;
    max-width: 1400px;
    margin: 0 auto;
}}
h1 {{ font-size: 1.8rem; margin-bottom: 8px; color: #f1f5f9; }}
h2 {{ font-size: 1.3rem; margin: 32px 0 16px; color: #f1f5f9; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
h3 {{ font-size: 1.1rem; margin: 16px 0 8px; color: #cbd5e1; }}
h4 {{ font-size: 0.9rem; margin: 12px 0 6px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
p {{ margin-bottom: 8px; }}
code {{ background: #2d2d44; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; }}
pre {{ background: #16162a; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.82em; white-space: pre-wrap; word-break: break-word; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.muted {{ color: var(--text-muted); font-style: italic; }}
.header-row {{ display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 12px; }}
.gen-time {{ color: var(--text-muted); font-size: 0.85rem; }}

/* Dashboard */
.dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 20px 0; }}
.stat-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}}
.stat-card .stat-value {{ font-size: 2rem; font-weight: 700; }}
.stat-card .stat-label {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }}

/* Badges */
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    vertical-align: middle;
}}
.dedup-badge {{
    display: inline-block;
    background: #6366f1;
    color: #fff;
    font-size: 0.65rem;
    padding: 1px 6px;
    border-radius: 10px;
    font-weight: 600;
    vertical-align: middle;
}}

/* Timeline */
.timeline-svg {{ width: 100%; height: auto; border-radius: 8px; }}
.evt-dot {{ cursor: pointer; }}
.evt-dot:hover {{ r: 7; opacity: 1; }}

/* Cards */
details.evidence-card, details.incident-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 8px;
    overflow: hidden;
    transition: border-color 0.2s;
}}
details.evidence-card:hover, details.incident-card:hover {{
    border-color: #444;
}}
details.sev-critical {{ border-left: 3px solid #dc2626; }}
details.sev-warning {{ border-left: 3px solid #f59e0b; }}
details.sev-info {{ border-left: 3px solid #3b82f6; }}
summary {{
    padding: 10px 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 0.88rem;
}}
summary::-webkit-details-marker {{ display: none; }}
summary::before {{ content: '\u25b6'; font-size: 0.7em; color: var(--text-muted); transition: transform 0.2s; }}
details[open] > summary::before {{ transform: rotate(90deg); }}
.card-body {{ padding: 12px 16px 16px; border-top: 1px solid var(--border); }}
.ep-id {{ font-family: monospace; color: var(--text-muted); font-size: 0.8rem; }}
.ep-cat {{ color: #a78bfa; font-size: 0.82rem; }}
.ep-ts {{ color: var(--text-muted); font-size: 0.78rem; font-family: monospace; }}
.ep-bag {{ color: #6ee7b7; font-size: 0.75rem; font-family: monospace; }}
.cat-icon {{ font-size: 1.1rem; }}

.log-msg {{ background: #16162a; padding: 8px 12px; border-radius: 4px; font-family: monospace; font-size: 0.82rem; margin-bottom: 8px; word-break: break-all; }}
.state-ctx {{ margin: 8px 0; font-size: 0.82rem; color: var(--text-muted); }}
.summary-pre {{ margin-top: 8px; font-size: 0.8rem; color: #cbd5e1; }}

/* Cross-validation table */
.cv-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 6px; }}
.cv-table th {{ text-align: left; padding: 4px 8px; color: var(--text-muted); border-bottom: 1px solid var(--border); }}
.cv-table td {{ padding: 4px 8px; border-bottom: 1px solid #222; }}

/* Incident-specific */
.incident-meta {{ display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px; }}
.root-cause {{ background: #1e1b4b; border: 1px solid #312e81; border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 0.88rem; }}
.suppressed-notice {{ background: #451a03; border: 1px solid #92400e; border-radius: 6px; padding: 8px 12px; margin: 8px 0; font-size: 0.82rem; color: #fbbf24; }}
.actions ul {{ padding-left: 20px; font-size: 0.85rem; }}
.actions li {{ margin: 4px 0; }}
.evidence-list {{ margin-top: 8px; }}
.evidence-item {{ font-size: 0.78rem; font-family: monospace; padding: 4px 8px; border-left: 2px solid #444; margin: 4px 0; word-break: break-all; }}
.evidence-item.log-ev {{ border-left-color: #8b5cf6; }}
.evidence-item.sensor-ev {{ border-left-color: #06b6d4; }}

/* Anomaly rows (mission-mode timeline) */
.anomaly-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
    flex-wrap: wrap;
}}
.anomaly-row:hover {{ background: #1a1a2e; }}
.anom-ts {{ font-family: monospace; color: var(--text-muted); font-size: 0.75rem; min-width: 160px; }}
.anom-cat {{ color: #a78bfa; min-width: 140px; }}
.anom-desc {{ flex: 1; }}
.anom-bag {{ color: #6ee7b7; font-family: monospace; font-size: 0.72rem; }}
.detail-chips {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 2px; }}
.detail-chip {{ background: #2d2d44; padding: 1px 6px; border-radius: 3px; font-size: 0.72rem; color: var(--text-muted); }}

/* Filter controls */
.filter-bar {{
    display: flex;
    gap: 8px;
    margin: 12px 0;
    flex-wrap: wrap;
    align-items: center;
}}
.filter-bar label {{ font-size: 0.82rem; color: var(--text-muted); }}
.filter-btn {{
    padding: 4px 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--text);
    font-size: 0.78rem;
    cursor: pointer;
}}
.filter-btn:hover {{ border-color: var(--accent); }}
.filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
input.search-input {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 10px;
    color: var(--text);
    font-size: 0.82rem;
    width: 220px;
}}

/* Foxglove layout section */
.foxglove-section {{ margin: 16px 0; }}
.layout-link {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 14px;
    margin: 4px;
    color: #38bdf8;
    font-size: 0.85rem;
    text-decoration: none;
    transition: background 0.2s;
}}
.layout-link:hover {{ background: #334155; text-decoration: none; }}

/* Section tabs */
.tab-bar {{ display: flex; gap: 4px; margin: 16px 0 0; }}
.tab-btn {{
    padding: 8px 20px;
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    background: transparent;
    color: var(--text-muted);
    font-size: 0.85rem;
    cursor: pointer;
}}
.tab-btn.active {{ background: var(--card-bg); color: var(--text); border-bottom: 2px solid var(--accent); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

@media (max-width: 768px) {{
    body {{ padding: 12px; }}
    .dashboard {{ grid-template-columns: repeat(2, 1fr); }}
    summary {{ font-size: 0.8rem; }}
}}
</style>
</head>
<body>

<div class="header-row">
    <h1>Robot Diagnostic Report</h1>
    <span class="gen-time">{generated_at}</span>
</div>
<p class="muted">{subtitle}</p>

<!-- Dashboard -->
<div class="dashboard">
{dashboard_cards}
</div>

<!-- Timeline -->
<h2>Timeline</h2>
{timeline_svg}

<!-- Foxglove Layouts -->
{foxglove_section}

<!-- Tab bar -->
<div class="tab-bar">
{tab_buttons}
</div>

<!-- Tab contents -->
{tab_contents}

<script>
// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
    }});
}});

// Severity filter
document.querySelectorAll('.filter-btn[data-severity]').forEach(btn => {{
    btn.addEventListener('click', () => {{
        btn.classList.toggle('active');
        applyFilters();
    }});
}});

// Search filter
const searchInput = document.querySelector('.search-input');
if (searchInput) {{
    searchInput.addEventListener('input', applyFilters);
}}

function applyFilters() {{
    const activeFilters = Array.from(document.querySelectorAll('.filter-btn.active[data-severity]'))
        .map(b => b.dataset.severity);
    const searchTerm = (document.querySelector('.search-input')?.value || '').toLowerCase();
    const cards = document.querySelectorAll('.evidence-card, .incident-card, .anomaly-row');

    cards.forEach(card => {{
        let show = true;
        if (activeFilters.length > 0) {{
            const cardSev = card.className.match(/sev-(\\w+)/)?.[1]?.toUpperCase() || '';
            show = activeFilters.includes(cardSev);
        }}
        if (show && searchTerm) {{
            show = card.textContent.toLowerCase().includes(searchTerm);
        }}
        card.style.display = show ? '' : 'none';
    }});
}}
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_html_report(
    cross_validation_report: Optional[str] = None,
    diagnostic_report: Optional[str] = None,
    foxglove_layouts: Optional[Dict[str, str]] = None,
    output_path: str = "diagnostic_report.html",
) -> str:
    """
    Generate a self-contained HTML diagnostic report.

    Args:
        cross_validation_report: Path to cross_validation_report.json
        diagnostic_report: Path to diagnostic_report.json (from analyze.py)
        foxglove_layouts: Dict of {category_name: layout_file_path}
        output_path: Where to write the HTML file

    Returns:
        Path to the generated HTML file
    """
    cv_data = None
    diag_data = None

    if cross_validation_report and os.path.exists(cross_validation_report):
        with open(cross_validation_report) as f:
            cv_data = json.load(f)

    if diagnostic_report and os.path.exists(diagnostic_report):
        with open(diagnostic_report) as f:
            diag_data = json.load(f)

    if not cv_data and not diag_data:
        raise ValueError("At least one of cross_validation_report or diagnostic_report must be provided")

    # --- Build dashboard ---
    dashboard_cards = []
    subtitle_parts = []
    all_timeline_events = []

    if cv_data:
        stats = cv_data.get("stats", {})
        verdicts = stats.get("verdicts", {})
        packets = cv_data.get("evidence_packets", [])

        dashboard_cards.extend([
            _stat_card(str(stats.get("total_bags", 0)), "Bags Analyzed"),
            _stat_card(str(stats.get("raw_log_events", 0)), "Raw Log Events"),
            _stat_card(str(stats.get("denoised_events", 0)), "After Denoising"),
            _stat_card(str(stats.get("evidence_packets", 0)), "Evidence Packets"),
            _stat_card(str(verdicts.get("CONFIRMED", 0)), "Confirmed", "#16a34a"),
            _stat_card(str(verdicts.get("CONTRADICTED", 0)), "Contradicted", "#dc2626"),
            _stat_card(f'{stats.get("elapsed_sec", 0)}s', "Analysis Time"),
        ])

        denoise = cv_data.get("denoise_stats", {})
        noise_pct = denoise.get("noise_reduction_pct", 0)
        subtitle_parts.append(f"Cross-Validation: {noise_pct:.0f}% noise filtered")

        # Build timeline events from packets
        for p in packets:
            if p.get("severity") in ("CRITICAL", "WARNING") or p.get("verdict") == "CONTRADICTED":
                ts = p.get("timestamp")
                # Try to parse timestamp string to float if needed
                if isinstance(ts, str):
                    try:
                        ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").replace(
                            tzinfo=CST).timestamp()
                    except (ValueError, TypeError):
                        ts = 0
                all_timeline_events.append({
                    "timestamp": ts,
                    "timestamp_str": p.get("timestamp", ""),
                    "severity": p.get("severity", "INFO"),
                    "category": p.get("category", ""),
                    "description": p.get("summary", ""),
                })

    if diag_data:
        missions = diag_data.get("missions", [])
        mode = diag_data.get("analysis_mode", "")
        total_anomalies = 0
        critical_count = 0
        warning_count = 0

        for mission in missions:
            timeline = mission.get("timeline", [])
            incidents = mission.get("incidents", [])
            items = timeline or incidents
            for item in items:
                total_anomalies += 1
                sev = item.get("severity", "INFO")
                if sev == "CRITICAL":
                    critical_count += 1
                elif sev == "WARNING":
                    warning_count += 1

                ts = item.get("timestamp", 0)
                if isinstance(ts, (int, float)) and ts > 0:
                    all_timeline_events.append(item)

        health_statuses = [m.get("overall_health", "UNKNOWN") for m in missions]
        unhealthy = sum(1 for h in health_statuses if h == "UNHEALTHY")

        dashboard_cards.extend([
            _stat_card(str(len(missions)), "Missions"),
            _stat_card(str(total_anomalies), "Total Anomalies"),
            _stat_card(str(critical_count), "Critical", "#dc2626"),
            _stat_card(str(warning_count), "Warnings", "#f59e0b"),
            _stat_card(f"{unhealthy}/{len(missions)}", "Unhealthy", "#dc2626" if unhealthy else "#16a34a"),
        ])
        subtitle_parts.append(f"Analyzer v{diag_data.get('analyzer_version', '?')} — {mode} mode")

    generated_at = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S CST")
    subtitle = " | ".join(subtitle_parts) if subtitle_parts else ""

    # --- Build timeline SVG ---
    timeline_svg = _build_timeline_svg(all_timeline_events)

    # --- Build Foxglove section ---
    foxglove_section = ""
    if foxglove_layouts:
        links = []
        for name, path in foxglove_layouts.items():
            fname = os.path.basename(path)
            links.append(
                f'<a class="layout-link" href="{_esc(fname)}" download>'
                f'\U0001f4ca {_esc(name)}</a>'
            )
        foxglove_section = f'''
        <h2>Foxglove Layouts</h2>
        <p class="muted">Click to download problem-specific Foxglove layouts. Import them in Foxglove Studio via Layout &gt; Import.</p>
        <div class="foxglove-section">{"".join(links)}</div>'''

    # --- Build tab contents ---
    tab_buttons = []
    tab_contents = []

    if cv_data:
        packets = cv_data.get("evidence_packets", [])

        # Contradictions tab
        contradictions = [p for p in packets if p.get("verdict") == "CONTRADICTED"]
        if contradictions:
            tab_buttons.append(_tab_button("contradictions", f"Contradictions ({len(contradictions)})", True))
            cards_html = "\n".join(_evidence_packet_card(p, i) for i, p in enumerate(contradictions))
            tab_contents.append(_tab_panel("contradictions", f'''
                <p class="muted">Log events where sensor data DISAGREES with the log claim — these are the most interesting findings.</p>
                {_filter_bar()}
                {cards_html}
            ''', True))

        # Critical tab
        criticals = [p for p in packets if p.get("severity") == "CRITICAL" and p.get("verdict") != "CONTRADICTED"]
        if criticals:
            is_first = not contradictions
            tab_buttons.append(_tab_button("criticals", f"Critical ({len(criticals)})", is_first))
            cards_html = "\n".join(_evidence_packet_card(p, i) for i, p in enumerate(criticals))
            tab_contents.append(_tab_panel("criticals", f'''
                {_filter_bar()}
                {cards_html}
            ''', is_first))

        # All packets tab
        tab_buttons.append(_tab_button("all-packets", f"All Packets ({len(packets)})", not contradictions and not criticals))
        cards_html = "\n".join(_evidence_packet_card(p, i) for i, p in enumerate(packets))
        tab_contents.append(_tab_panel("all-packets", f'''
            {_filter_bar()}
            {cards_html}
        ''', not contradictions and not criticals))

        # State timeline tab
        state_summary = cv_data.get("state_timeline_summary", {})
        if state_summary:
            tab_buttons.append(_tab_button("state-timeline", "State Timeline", False))
            state_rows = []
            for key, info in state_summary.items():
                state_rows.append(
                    f'<tr><td><code>{_esc(key)}</code></td>'
                    f'<td>{info.get("total_transitions", 0)}</td>'
                    f'<td><code>{_esc(str(info.get("current_value", "")))}</code></td></tr>'
                )
            tab_contents.append(_tab_panel("state-timeline", f'''
                <table class="cv-table">
                    <tr><th>State Key</th><th>Transitions</th><th>Current Value</th></tr>
                    {"".join(state_rows)}
                </table>
            ''', False))

    if diag_data:
        missions = diag_data.get("missions", [])
        for mi, mission in enumerate(missions):
            mid = mission.get("mission_id", mi + 1)
            health = mission.get("overall_health", "UNKNOWN")
            timeline = mission.get("timeline", [])
            incidents = mission.get("incidents", [])

            health_color = "#dc2626" if health == "UNHEALTHY" else "#16a34a"
            label = f'Mission {mid} <span class="badge" style="background:{health_color}">{health}</span>'
            is_first_tab = (mi == 0 and not cv_data)

            tab_buttons.append(_tab_button(f"mission-{mid}", label, is_first_tab))

            body_parts = [
                f'<div class="incident-meta">'
                f'<span>Duration: <strong>{mission.get("duration_sec", 0):.0f}s</strong></span>'
                f'<span>Bags: <strong>{mission.get("num_bags", 0)}</strong></span>'
                f'<span>Time: {_esc(mission.get("start_time", ""))} → {_esc(mission.get("end_time", ""))}</span>'
                f'</div>',
            ]

            if incidents:
                body_parts.append("<h3>Incidents</h3>")
                body_parts.append(_filter_bar())
                for ii, inc in enumerate(incidents):
                    body_parts.append(_incident_card(inc, ii))

            if timeline:
                body_parts.append(f"<h3>Anomaly Timeline ({len(timeline)} events)</h3>")
                if not incidents:
                    body_parts.append(_filter_bar())
                # Group by severity for easier scanning
                for sev in ["CRITICAL", "WARNING", "INFO"]:
                    sev_events = [a for a in timeline if a.get("severity") == sev]
                    if sev_events:
                        body_parts.append(f"<h4>{sev} ({len(sev_events)})</h4>")
                        for a in sev_events:
                            body_parts.append(_anomaly_card(a))

            tab_contents.append(_tab_panel(f"mission-{mid}", "\n".join(body_parts), is_first_tab))

    # --- Assemble ---
    html = HTML_TEMPLATE.format(
        generated_at=generated_at,
        subtitle=subtitle,
        dashboard_cards="\n".join(dashboard_cards),
        timeline_svg=timeline_svg,
        foxglove_section=foxglove_section,
        tab_buttons="\n".join(tab_buttons),
        tab_contents="\n".join(tab_contents),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report generated: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _stat_card(value: str, label: str, color: str = "#3b82f6") -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="stat-value" style="color:{color}">{_esc(value)}</div>'
        f'<div class="stat-label">{_esc(label)}</div>'
        f'</div>'
    )


def _tab_button(tab_id: str, label: str, active: bool = False) -> str:
    cls = "tab-btn active" if active else "tab-btn"
    return f'<button class="{cls}" data-tab="tab-{tab_id}">{label}</button>'


def _tab_panel(tab_id: str, content: str, active: bool = False) -> str:
    cls = "tab-content active" if active else "tab-content"
    return f'<div class="{cls}" id="tab-{tab_id}">{content}</div>'


def _filter_bar() -> str:
    return '''
    <div class="filter-bar">
        <label>Filter:</label>
        <button class="filter-btn" data-severity="CRITICAL">Critical</button>
        <button class="filter-btn" data-severity="WARNING">Warning</button>
        <button class="filter-btn" data-severity="INFO">Info</button>
        <input class="search-input" type="text" placeholder="Search...">
    </div>'''


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate HTML diagnostic report")
    parser.add_argument("--cross-validation", "-cv", help="Path to cross_validation_report.json")
    parser.add_argument("--diagnostic", "-d", help="Path to diagnostic_report.json")
    parser.add_argument("--output", "-o", default="diagnostic_report.html", help="Output HTML path")
    args = parser.parse_args()

    generate_html_report(
        cross_validation_report=args.cross_validation,
        diagnostic_report=args.diagnostic,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
