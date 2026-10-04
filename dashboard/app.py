"""
dashboard/app.py – Real-time web dashboard for QuantForge.

Serves a single-page app showing:
- Leaderboard with robustness scores
- Trial history (recent trials, pass/fail rates)
- Live metrics for surviving strategies
- Gate breakdown per strategy

Uses Python's built-in http.server + JSON API endpoints.
No external web framework required.

Run:  python -m dashboard.app
  or: python main.py dashboard
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config, load_env
from storage.mongo import get_db


class DashboardHandler(SimpleHTTPRequestHandler):
    """Handle API requests and serve static files."""

    cfg = None
    db = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self._handle_api(path, parse_qs(parsed.query))
        elif path == "/" or path == "/index.html":
            self._serve_html()
        else:
            self.send_error(404)

    def _handle_api(self, path: str, params: dict):
        try:
            if path == "/api/leaderboard":
                data = self._get_leaderboard()
            elif path == "/api/trials":
                limit = int(params.get("limit", [50])[0])
                data = self._get_trials(limit)
            elif path == "/api/stats":
                data = self._get_stats()
            elif path == "/api/trial":
                trial_id = params.get("id", [""])[0]
                data = self._get_trial_detail(trial_id)
            else:
                self.send_error(404)
                return
            self._json_response(data)
        except Exception as e:
            self._json_response({"error": str(e)}, status=500)

    def _json_response(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _get_leaderboard(self):
        entries = list(self.db["leaderboard"].find(
            {}, {"_id": 0}
        ).sort("robustness_score", -1).limit(50))
        return {"entries": entries}

    def _get_trials(self, limit=50):
        trials = list(self.db["trials"].find(
            {}, {"_id": 0, "source_code": 0, "train_result.trades": 0,
                 "val_result.trades": 0, "train_result.equity_curve_summary": 0}
        ).sort("started_at", -1).limit(limit))
        return {"trials": trials}

    def _get_stats(self):
        total = self.db["trials"].count_documents({})
        survived = self.db["trials"].count_documents({"status": "survived"})
        rejected = self.db["trials"].count_documents({"status": "rejected"})
        errors = self.db["trials"].count_documents({"status": "error"})

        # Rejection breakdown
        pipeline = [
            {"$match": {"status": "rejected"}},
            {"$group": {"_id": "$rejected_at", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        rejection_breakdown = list(self.db["trials"].aggregate(pipeline))

        return {
            "total_trials": total,
            "survived": survived,
            "rejected": rejected,
            "errors": errors,
            "survival_rate": round(survived / max(total, 1) * 100, 1),
            "rejection_breakdown": [
                {"gate": r["_id"], "count": r["count"]}
                for r in rejection_breakdown
            ],
        }

    def _get_trial_detail(self, trial_id: str):
        trial = self.db["trials"].find_one(
            {"trial_id": trial_id}, {"_id": 0}
        )
        if not trial:
            return {"error": "Trial not found"}
        return trial

    def _serve_html(self):
        html = DASHBOARD_HTML
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Suppress request logs


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuantForge Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2236;
    --bg-card-hover: #1f2a40;
    --border: #2a3654;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-green: #10b981;
    --accent-green-dim: rgba(16, 185, 129, 0.15);
    --accent-red: #ef4444;
    --accent-red-dim: rgba(239, 68, 68, 0.15);
    --accent-blue: #3b82f6;
    --accent-blue-dim: rgba(59, 130, 246, 0.15);
    --accent-yellow: #f59e0b;
    --accent-yellow-dim: rgba(245, 158, 11, 0.15);
    --accent-purple: #8b5cf6;
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    --radius: 12px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
}

.header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
}

.header h1 {
    font-size: 20px;
    font-weight: 700;
    background: var(--gradient-2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.header .status {
    font-size: 13px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 32px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    transition: all 0.2s ease;
}

.stat-card:hover {
    background: var(--bg-card-hover);
    transform: translateY(-2px);
    box-shadow: var(--shadow);
}

.stat-card .label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.stat-card .value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.stat-card.green .value { color: var(--accent-green); }
.stat-card.red .value { color: var(--accent-red); }
.stat-card.blue .value { color: var(--accent-blue); }
.stat-card.yellow .value { color: var(--accent-yellow); }

.section {
    margin-bottom: 32px;
}

.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}

.section-header h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
}

.section-header .badge {
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 600;
    background: var(--accent-blue-dim);
    color: var(--accent-blue);
}

table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
}

th {
    text-align: left;
    padding: 12px 16px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
}

td {
    padding: 12px 16px;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
}

tr:last-child td { border-bottom: none; }

tr:hover td { background: var(--bg-card-hover); }

.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}

.pill.survived { background: var(--accent-green-dim); color: var(--accent-green); }
.pill.rejected { background: var(--accent-red-dim); color: var(--accent-red); }
.pill.error { background: var(--accent-yellow-dim); color: var(--accent-yellow); }

.score-bar {
    width: 80px;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
}

.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--gradient-2);
    transition: width 0.5s ease;
}

.rejection-chart {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.rejection-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    min-width: 140px;
    text-align: center;
}

.rejection-bar .gate-name {
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 4px;
}

.rejection-bar .gate-count {
    font-size: 20px;
    font-weight: 700;
    color: var(--accent-red);
    font-family: 'JetBrains Mono', monospace;
}

.empty-state {
    text-align: center;
    padding: 48px;
    color: var(--text-muted);
    font-size: 14px;
}

.empty-state .icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.3;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-in {
    animation: fadeIn 0.3s ease forwards;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.live-dot {
    width: 8px;
    height: 8px;
    background: var(--accent-green);
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
    margin-right: 6px;
}

@media (max-width: 768px) {
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .container { padding: 16px; }
    .header { padding: 12px 16px; }
}
</style>
</head>
<body>

<div class="header">
    <h1>QuantForge</h1>
    <div class="status">
        <span class="live-dot"></span>
        <span id="last-update">Loading...</span>
    </div>
</div>

<div class="container">
    <!-- Stats -->
    <div class="stats-grid" id="stats-grid">
        <div class="stat-card blue"><div class="label">Total Trials</div><div class="value" id="stat-total">-</div></div>
        <div class="stat-card green"><div class="label">Survived</div><div class="value" id="stat-survived">-</div></div>
        <div class="stat-card red"><div class="label">Rejected</div><div class="value" id="stat-rejected">-</div></div>
        <div class="stat-card yellow"><div class="label">Survival Rate</div><div class="value" id="stat-rate">-</div></div>
    </div>

    <!-- Rejection Breakdown -->
    <div class="section">
        <div class="section-header">
            <h2>Rejection Breakdown</h2>
        </div>
        <div class="rejection-chart" id="rejection-chart"></div>
    </div>

    <!-- Leaderboard -->
    <div class="section">
        <div class="section-header">
            <h2>Leaderboard</h2>
            <span class="badge" id="lb-count">0 strategies</span>
        </div>
        <div id="leaderboard-content"></div>
    </div>

    <!-- Recent Trials -->
    <div class="section">
        <div class="section-header">
            <h2>Recent Trials</h2>
            <span class="badge" id="trial-count">0 trials</span>
        </div>
        <div id="trials-content"></div>
    </div>
</div>

<script>
const API_BASE = '';

async function fetchJSON(url) {
    const resp = await fetch(API_BASE + url);
    return resp.json();
}

function formatTime(ts) {
    if (!ts) return '-';
    const d = new Date(ts);
    return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function renderStats(stats) {
    document.getElementById('stat-total').textContent = stats.total_trials;
    document.getElementById('stat-survived').textContent = stats.survived;
    document.getElementById('stat-rejected').textContent = stats.rejected;
    document.getElementById('stat-rate').textContent = stats.survival_rate + '%';

    const chart = document.getElementById('rejection-chart');
    if (stats.rejection_breakdown.length === 0) {
        chart.innerHTML = '<div class="empty-state">No rejections yet</div>';
        return;
    }
    chart.innerHTML = stats.rejection_breakdown.map(r => `
        <div class="rejection-bar animate-in">
            <div class="gate-name">${r.gate || 'unknown'}</div>
            <div class="gate-count">${r.count}</div>
        </div>
    `).join('');
}

function renderLeaderboard(data) {
    document.getElementById('lb-count').textContent = `${data.entries.length} strategies`;
    if (data.entries.length === 0) {
        document.getElementById('leaderboard-content').innerHTML =
            '<div class="empty-state"><div class="icon">&#x1F3C6;</div>No survivors yet. Strategies must pass all gates to appear here.</div>';
        return;
    }
    const rows = data.entries.map((e, i) => `
        <tr class="animate-in" style="animation-delay: ${i * 50}ms">
            <td style="color: ${i < 3 ? 'var(--accent-yellow)' : 'var(--text-secondary)'}">#${i + 1}</td>
            <td style="font-family: Inter, sans-serif; font-weight: 500; color: var(--text-primary)">${e.strategy_name || '?'}</td>
            <td>${e.timeframe || '?'}</td>
            <td>
                <div class="score-bar"><div class="score-bar-fill" style="width: ${(e.robustness_score || 0) * 100}%"></div></div>
                ${(e.robustness_score || 0).toFixed(4)}
            </td>
            <td style="color: ${(e.train_sharpe || 0) > 1 ? 'var(--accent-green)' : 'var(--text-secondary)'}">${(e.train_sharpe || 0).toFixed(3)}</td>
            <td style="color: ${(e.val_sharpe || 0) > 0.5 ? 'var(--accent-green)' : 'var(--text-secondary)'}">${(e.val_sharpe || 0).toFixed(3)}</td>
            <td>${e.train_trades || 0}</td>
        </tr>
    `).join('');
    document.getElementById('leaderboard-content').innerHTML = `
        <table>
            <thead><tr><th>Rank</th><th>Strategy</th><th>TF</th><th>Robustness</th><th>Train Sharpe</th><th>Val Sharpe</th><th>Trades</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function renderTrials(data) {
    document.getElementById('trial-count').textContent = `${data.trials.length} trials`;
    if (data.trials.length === 0) {
        document.getElementById('trials-content').innerHTML =
            '<div class="empty-state"><div class="icon">&#x1F52C;</div>No trials yet. Run: python main.py run --trials 10</div>';
        return;
    }
    const rows = data.trials.map((t, i) => {
        const metrics = t.train_result?.metrics || {};
        return `
        <tr class="animate-in" style="animation-delay: ${i * 30}ms">
            <td style="font-family: Inter, sans-serif; font-weight: 500; color: var(--text-primary)">${t.strategy_name || '?'}</td>
            <td>${t.timeframe || '?'}</td>
            <td><span class="pill ${t.status}">${t.status}</span></td>
            <td style="color: var(--text-muted)">${t.rejected_at || '-'}</td>
            <td>${(metrics.sharpe || 0).toFixed(3)}</td>
            <td>${(metrics.profit_factor || 0).toFixed(2)}</td>
            <td>${metrics.total_trades || 0}</td>
            <td style="color: var(--text-muted)">${(t.wall_seconds || 0).toFixed(1)}s</td>
            <td style="color: var(--text-muted)">${formatTime(t.started_at)}</td>
        </tr>`;
    }).join('');
    document.getElementById('trials-content').innerHTML = `
        <table>
            <thead><tr><th>Strategy</th><th>TF</th><th>Status</th><th>Rejected At</th><th>Sharpe</th><th>PF</th><th>Trades</th><th>Time</th><th>Date</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

async function refresh() {
    try {
        const [stats, lb, trials] = await Promise.all([
            fetchJSON('/api/stats'),
            fetchJSON('/api/leaderboard'),
            fetchJSON('/api/trials?limit=30'),
        ]);
        renderStats(stats);
        renderLeaderboard(lb);
        renderTrials(trials);
        document.getElementById('last-update').textContent = 'Updated ' + new Date().toLocaleTimeString();
    } catch (e) {
        document.getElementById('last-update').textContent = 'Error: ' + e.message;
    }
}

refresh();
setInterval(refresh, 5000);
</script>

</body>
</html>"""


def run_dashboard(cfg: dict | None = None):
    """Start the dashboard server."""
    if cfg is None:
        load_env()
        cfg = load_config()

    host = cfg.get("dashboard", {}).get("web_host", "127.0.0.1")
    port = cfg.get("dashboard", {}).get("web_port", 8000)

    DashboardHandler.cfg = cfg
    DashboardHandler.db = get_db(cfg.get("mongo", {}).get("database", "quantforge"))

    server = HTTPServer((host, port), DashboardHandler)
    print(f"QuantForge Dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    run_dashboard()
