/**
 * QuantForge Dashboard Application (ES Modules, Zero CDN Runtime)
 */

let state = {
    csrfToken: '',
    currentTab: 'overview',
    currentSubtab: 'equity',
    engineState: 'STOPPED',
    sseConnected: false,
    selectedStrategyId: null,
    logAutoScroll: true,
    leaderboardItems: [],
};

// ─── API Client with CSRF ──────────────────────────────────────────────────
async function api(path, options = {}) {
    const headers = options.headers || {};
    if (state.csrfToken) {
        headers['X-CSRF-Token'] = state.csrfToken;
    }
    if (options.body && typeof options.body === 'object') {
        headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(options.body);
    }
    options.headers = headers;

    const resp = await fetch(path, options);
    if (resp.status === 401) {
        showLoginModal();
        throw new Error('Unauthorized');
    }
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Request failed');
    }
    return resp.json();
}

// ─── Tab Navigation ────────────────────────────────────────────────────────
function setupNavigation() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const pageId = tab.dataset.page;
            showPage(pageId);
        });
    });

    document.querySelectorAll('.strat-subtab').forEach(tab => {
        tab.addEventListener('click', () => {
            const sub = tab.dataset.subtab;
            showStrategySubtab(sub);
        });
    });

    document.getElementById('split-select')?.addEventListener('change', () => {
        loadLeaderboard();
    });

    document.getElementById('filter-timeframe')?.addEventListener('change', () => {
        loadLeaderboard();
    });

    document.getElementById('leaderboard-search')?.addEventListener('input', (e) => {
        renderLeaderboardRows(e.target.value.trim().toLowerCase());
    });

    document.getElementById('btn-export-csv')?.addEventListener('click', exportLeaderboardCSV);

    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);

    document.getElementById('auth-status-btn')?.addEventListener('click', showLoginModal);

    document.getElementById('btn-pause-log')?.addEventListener('click', (e) => {
        state.logAutoScroll = !state.logAutoScroll;
        e.target.innerText = state.logAutoScroll ? 'Pause Auto-Scroll' : 'Resume Auto-Scroll';
    });

    document.getElementById('btn-clear-logs')?.addEventListener('click', () => {
        const win = document.getElementById('log-stream-window');
        if (win) win.innerHTML = '<div style="color: var(--text-muted);">Log cleared.</div>';
    });

    document.getElementById('log-filter-level')?.addEventListener('change', () => {
        loadLogs();
    });

    document.getElementById('settings-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        await saveSettings();
    });
}

function showPage(pageId) {
    state.currentTab = pageId;
    document.querySelectorAll('.nav-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.page === pageId);
    });
    document.querySelectorAll('.page-view').forEach(p => {
        p.classList.toggle('active', p.id === `page-${pageId}`);
    });

    if (pageId === 'leaderboard') loadLeaderboard();
    if (pageId === 'overview') { loadFunnel(); loadActivityOverview(); }
    if (pageId === 'llm') loadLLMStats();
    if (pageId === 'logs') loadLogs();
    if (pageId === 'coverage') loadCoverageMap();
    if (pageId === 'data') loadDataIntegrity();
    if (pageId === 'settings') loadSettings();
}

function showStrategySubtab(subId) {
    state.currentSubtab = subId;
    document.querySelectorAll('.strat-subtab').forEach(t => {
        t.classList.toggle('active', t.dataset.subtab === subId);
    });

    const views = ['equity', 'trades', 'gates', 'code', 'llm'];
    views.forEach(v => {
        const el = document.getElementById(`subtab-${v}-view`);
        if (el) el.style.display = (v === subId) ? 'block' : 'none';
    });
}

function toggleTheme() {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = (cur === 'dark') ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('qf_theme', next); } catch (e) {}
}

// ─── Engine State & SSE Stream ─────────────────────────────────────────────
function initSSE() {
    const evtSource = new EventSource('/api/stream');

    evtSource.onmessage = function (event) {
        state.sseConnected = true;
        try {
            const data = JSON.parse(event.data);
            updateEngineUI(data);
        } catch (e) {
            console.error('Error parsing SSE data', e);
        }
    };

    evtSource.onerror = function () {
        state.sseConnected = false;
        evtSource.close();
        setTimeout(initSSE, 3000);
    };
}

function updateEngineUI(data) {
    state.engineState = data.state;
    const pill = document.getElementById('engine-state-pill');
    if (pill) {
        pill.className = `state-pill ${data.state}`;
        pill.innerHTML = `<span class="pulse-dot"></span> ${data.state}`;
    }

    const reasonEl = document.getElementById('engine-reason-text');
    if (reasonEl) reasonEl.innerText = data.reason || 'Active execution';

    const cycleElem = document.getElementById('engine-cycle-id');
    if (cycleElem) cycleElem.innerText = data.current_cycle_id || 'cycle_active';

    const hbElem = document.getElementById('engine-heartbeat-age');
    if (hbElem) hbElem.innerText = (data.heartbeat_age !== undefined) ? data.heartbeat_age.toFixed(1) : '0.0';

    const workerElem = document.getElementById('engine-worker-id');
    if (workerElem) workerElem.innerText = data.worker_id || 'worker-1';

    const trialsElem = document.getElementById('counter-total-trials');
    if (trialsElem) trialsElem.innerText = (data.total_trials || 0).toLocaleString();

    const candElem = document.getElementById('counter-candidates');
    if (candElem) candElem.innerText = (data.candidates_count || 0).toLocaleString();

    // Update control buttons
    const btnStart = document.getElementById('btn-start');
    const btnPause = document.getElementById('btn-pause');
    const btnStop = document.getElementById('btn-stop');
    const btnForce = document.getElementById('btn-force-stop');

    if (btnStart) btnStart.disabled = (data.state === 'RUNNING');
    if (btnPause) {
        btnPause.disabled = (data.state !== 'RUNNING' && data.state !== 'PAUSED');
        btnPause.innerText = (data.state === 'PAUSED') ? '▶ Resume' : '⏸ Pause';
    }
    if (btnStop) btnStop.disabled = (data.state === 'STOPPED');
    if (btnForce) btnForce.disabled = (data.state === 'STOPPED');
}

// ─── Engine Controls ───────────────────────────────────────────────────────
function setupControls() {
    document.getElementById('btn-start')?.addEventListener('click', async () => {
        try {
            await api('/api/engine/start', { method: 'POST' });
        } catch (e) { alert(e.message); }
    });

    document.getElementById('btn-pause')?.addEventListener('click', async () => {
        try {
            const action = (state.engineState === 'PAUSED') ? 'resume' : 'pause';
            await api(`/api/engine/${action}`, { method: 'POST' });
        } catch (e) { alert(e.message); }
    });

    document.getElementById('btn-stop')?.addEventListener('click', () => {
        showConfirmModal('Graceful Stop', 'Stop engine gracefully after finishing the current backtest trial?', async () => {
            await api('/api/engine/stop', { method: 'POST' });
        });
    });

    document.getElementById('btn-force-stop')?.addEventListener('click', () => {
        showConfirmModal('Force Stop (Immediate)', 'Kill engine process immediately? Any in-flight tasks will be aborted.', async () => {
            await api('/api/engine/force_stop', { method: 'POST' });
        });
    });
}

// ─── Funnel & Activity Charts ──────────────────────────────────────────────
async function loadFunnel() {
    try {
        const data = await api('/api/stats/funnel');
        const stages = (data.funnel || []).map(f => f.stage.replace(/_/g, ' '));
        const counts = (data.funnel || []).map(f => f.count);

        const canvas = document.getElementById('funnel-chart');
        if (canvas && window.Chart) {
            new window.Chart(canvas, {
                type: 'bar',
                data: {
                    labels: stages,
                    datasets: [{
                        label: 'Survivors by Stage',
                        data: counts,
                        backgroundColor: '#3b82f6',
                        borderRadius: 4,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                }
            });
        }

        const rejMap = data.rejections_by_stage || {};
        const lookaheadRej = (rejMap['determinism'] || 0) + (rejMap['truncation'] || 0) + (rejMap['delay'] || 0);
        const robustRej = (rejMap['monte_carlo'] || 0) + (rejMap['walk_forward'] || 0) + (rejMap['regime_and_year'] || 0);

        const lEl = document.getElementById('counter-lookahead-rej');
        if (lEl) lEl.innerText = lookaheadRej.toLocaleString();
        const rEl = document.getElementById('counter-robustness-rej');
        if (rEl) rEl.innerText = robustRej.toLocaleString();
    } catch (e) {
        console.error('Failed loading funnel', e);
    }
}

async function loadActivityOverview() {
    try {
        const data = await api('/api/leaderboard?limit=10');
        const items = data.items || [];
        const canvas = document.getElementById('activity-chart');
        if (canvas && window.Chart && items.length) {
            const labels = items.map(it => (it.strategy_id || it.name || '').slice(0, 16));
            const scores = items.map(it => it.robustness_score || 0);

            new window.Chart(canvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Robustness Score (Top 10)',
                        data: scores,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.3,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                }
            });
        }
    } catch (e) {
        console.error('Failed loading activity', e);
    }
}

// ─── Leaderboard ───────────────────────────────────────────────────────────
async function loadLeaderboard() {
    try {
        const split = document.getElementById('split-select')?.value || 'train';
        const tf = document.getElementById('filter-timeframe')?.value || '';
        let url = `/api/leaderboard?split=${split}&limit=50`;
        if (tf) url += `&timeframe=${tf}`;

        const data = await api(url);
        state.leaderboardItems = data.items || [];
        renderLeaderboardRows();
    } catch (e) {
        console.error('Failed loading leaderboard', e);
    }
}

function renderLeaderboardRows(filterText = '') {
    const tbody = document.getElementById('leaderboard-tbody');
    if (!tbody) return;

    let items = state.leaderboardItems;
    if (filterText) {
        items = items.filter(it =>
            (it.strategy_id || '').toLowerCase().includes(filterText) ||
            (it.concept_family || '').toLowerCase().includes(filterText)
        );
    }

    if (!items.length) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 32px;">No candidate strategies found matching query.</td></tr>`;
        return;
    }

    tbody.innerHTML = items.map(item => `
        <tr style="cursor: pointer;" data-id="${item.strategy_id || item.name}">
            <td><strong>${item.strategy_id || item.name}</strong></td>
            <td><span class="badge badge-amber">CANDIDATE (unproven)</span></td>
            <td>${item.timeframe || '1h'}</td>
            <td>${item.concept_family || 'trend'}</td>
            <td class="mono" style="font-weight: 700; color: var(--color-green);">${(item.robustness_score || 0).toFixed(1)}</td>
            <td class="mono">${(item.train_metrics?.sharpe || 0).toFixed(2)}</td>
            <td class="mono">${(item.train_metrics?.profit_factor || 0).toFixed(2)}</td>
            <td class="mono">${((item.train_metrics?.max_drawdown || 0) * 100).toFixed(1)}%</td>
            <td class="mono">${item.train_metrics?.total_trades || 0}</td>
        </tr>
    `).join('');

    tbody.querySelectorAll('tr').forEach(row => {
        row.addEventListener('click', () => {
            const sid = row.dataset.id;
            if (sid) {
                showPage('strategy');
                loadStrategyDetail(sid);
            }
        });
    });
}

function exportLeaderboardCSV() {
    const items = state.leaderboardItems;
    if (!items.length) return alert('No data to export.');

    const headers = ['Strategy ID', 'Status', 'Timeframe', 'Concept Family', 'Robustness Score', 'Sharpe', 'Profit Factor', 'Max Drawdown', 'Trades'];
    const rows = items.map(it => [
        it.strategy_id || it.name,
        'CANDIDATE',
        it.timeframe || '1h',
        it.concept_family || 'trend',
        (it.robustness_score || 0).toFixed(2),
        (it.train_metrics?.sharpe || 0).toFixed(2),
        (it.train_metrics?.profit_factor || 0).toFixed(2),
        ((it.train_metrics?.max_drawdown || 0) * 100).toFixed(2) + '%',
        it.train_metrics?.total_trades || 0,
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' +
        [headers.join(','), ...rows.map(e => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `quantforge_leaderboard_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ─── Strategy Detail ───────────────────────────────────────────────────────
async function loadStrategyDetail(stratId) {
    state.selectedStrategyId = stratId;
    document.getElementById('strat-detail-title').innerText = stratId;
    document.getElementById('strat-detail-subtitle').innerText = 'Loading strategy telemetry & validation metrics...';

    try {
        const detail = await api(`/api/strategies/${stratId}`);
        document.getElementById('strat-detail-subtitle').innerText = `Concept: ${detail.concept_family || 'trend'} | TF: ${detail.timeframe || '1h'} | Robustness: ${(detail.robustness_score || 0).toFixed(1)}`;

        // Source code
        const codePre = document.getElementById('strat-code-content');
        if (codePre) codePre.innerText = detail.source_code || detail.code || '# Source code not provided in summary.';

        // Load sub-endpoints
        loadStrategyEquity(stratId);
        loadStrategyTrades(stratId);
        loadStrategyGates(stratId);
        loadStrategyLLM(stratId);
    } catch (e) {
        console.error('Failed loading strategy detail', e);
    }
}

async function loadStrategyEquity(stratId) {
    try {
        const eqData = await api(`/api/strategies/${stratId}/equity`);
        const canvas = document.getElementById('strat-equity-chart');
        if (canvas && window.Chart) {
            const trainEq = eqData.train_equity || [];
            const valEq = eqData.val_equity || [];

            new window.Chart(canvas, {
                type: 'line',
                data: {
                    labels: trainEq.map((_, i) => i + 1),
                    datasets: [
                        {
                            label: 'Train In-Sample Equity',
                            data: trainEq,
                            borderColor: '#3b82f6',
                            tension: 0.1,
                            fill: false,
                        },
                        {
                            label: 'Validation Equity',
                            data: valEq,
                            borderColor: '#10b981',
                            tension: 0.1,
                            fill: false,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                }
            });
        }
    } catch (e) {
        console.error('Failed loading strategy equity', e);
    }
}

async function loadStrategyTrades(stratId) {
    try {
        const data = await api(`/api/strategies/${stratId}/trades`);
        const tbody = document.getElementById('strat-trades-tbody');
        if (!tbody) return;

        const trades = data.items || [];
        if (!trades.length) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">No trade execution records.</td></tr>';
            return;
        }

        tbody.innerHTML = trades.map((t, idx) => `
            <tr>
                <td class="mono">${idx + 1}</td>
                <td>${t.entry_time || '—'}</td>
                <td>${t.exit_time || '—'}</td>
                <td><span class="badge ${t.type === 'BUY' ? 'badge-green' : 'badge-red'}">${t.type || 'BUY'}</span></td>
                <td class="mono" style="color: ${(t.pnl || 0) >= 0 ? 'var(--color-green)' : 'var(--color-red)'}">$${(t.pnl || 0).toFixed(2)}</td>
                <td class="mono">${((t.return_pct || 0) * 100).toFixed(2)}%</td>
                <td>${t.exit_reason || 'take_profit'}</td>
            </tr>
        `).join('');
    } catch (e) {
        console.error('Failed loading strategy trades', e);
    }
}

async function loadStrategyGates(stratId) {
    try {
        const data = await api(`/api/strategies/${stratId}/gates`);
        const container = document.getElementById('strat-gates-content');
        if (!container) return;

        const gates = data.gate_results || {};
        const entries = Object.entries(gates);
        if (!entries.length) {
            container.innerHTML = '<div style="color: var(--text-muted);">No detailed gate reports stored.</div>';
            return;
        }

        container.innerHTML = entries.map(([gate, res]) => `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-bottom: 1px solid var(--border-subtle);">
                <span style="font-weight: 600;">${gate.replace(/_/g, ' ')}</span>
                <span class="badge ${res.passed ? 'badge-green' : 'badge-red'}">${res.passed ? 'PASS' : 'FAIL'}</span>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed loading strategy gates', e);
    }
}

async function loadStrategyLLM(stratId) {
    try {
        const data = await api(`/api/strategies/${stratId}/llm`);
        const container = document.getElementById('strat-llm-content');
        if (!container) return;

        const calls = data.calls || [];
        if (!calls.length) {
            container.innerHTML = '<div style="color: var(--text-muted);">No LLM calls mapped to this strategy.</div>';
            return;
        }

        container.innerHTML = calls.map(c => `
            <div style="margin-bottom: 16px; padding: 12px; background: var(--bg-base); border-radius: var(--radius-md);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <strong>Purpose: ${c.purpose || 'ideate'} (${c.model || 'model'})</strong>
                    <span class="mono">${c.total_tokens || 0} tokens</span>
                </div>
                <div style="font-size: 12px; color: var(--text-muted);">${c.timestamp || ''} | Key: ${c.key_label || 'key_1'} | Latency: ${c.latency_ms || 0}ms</div>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed loading strategy LLM info', e);
    }
}

// ─── LLM Usage ─────────────────────────────────────────────────────────────
async function loadLLMStats() {
    try {
        const data = await api('/api/llm/usage');
        const tok = data.summary?.total_tokens || 0;
        document.getElementById('llm-total-tokens').innerText = tok.toLocaleString();
        document.getElementById('llm-total-calls').innerText = (data.summary?.total_calls || 0).toLocaleString();
        document.getElementById('llm-failed-calls').innerText = (data.summary?.failed_calls || 0).toLocaleString();

        const tbody = document.getElementById('llm-calls-tbody');
        if (tbody) {
            const calls = data.recent_calls || [];
            if (!calls.length) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 24px;">No LLM calls logged yet.</td></tr>';
                return;
            }
            tbody.innerHTML = calls.map(c => `
                <tr>
                    <td class="mono">${c.timestamp ? new Date(c.timestamp * 1000).toISOString().slice(11, 19) : '—'}</td>
                    <td><span class="badge badge-blue">${c.key_label || 'key_1'}</span></td>
                    <td>${c.model || 'gemini-1.5-pro'}</td>
                    <td>${c.purpose || 'ideate'}</td>
                    <td>${c.strategy_id || '—'}</td>
                    <td class="mono">${(c.total_tokens || 0).toLocaleString()}</td>
                    <td class="mono">${c.latency_ms || 0}</td>
                    <td><span class="badge ${c.status === 'ok' ? 'badge-green' : 'badge-amber'}">${c.status || 'ok'}</span></td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error('Failed loading LLM stats', e);
    }
}

// ─── Live Log Viewer ───────────────────────────────────────────────────────
async function loadLogs() {
    try {
        const level = document.getElementById('log-filter-level')?.value || 'ALL';
        const data = await api(`/api/logs?level=${level}&limit=100`);
        const win = document.getElementById('log-stream-window');
        if (!win) return;

        const logs = data.logs || [];
        if (!logs.length) {
            win.innerHTML = '<div style="color: var(--text-muted);">No log records matching filter.</div>';
            return;
        }

        win.innerHTML = logs.map(l => {
            const color = l.level === 'CRITICAL' || l.level === 'ERROR' ? 'var(--color-red)' :
                l.level === 'WARNING' ? 'var(--color-amber)' : 'var(--text-secondary)';
            return `<div style="color: ${color}; margin-bottom: 2px;">
                <span class="mono" style="color: var(--text-muted);">${l.timestamp ? new Date(l.timestamp * 1000).toISOString().slice(11, 19) : ''}</span>
                <strong>[${l.level || 'INFO'}]</strong>
                <span>${l.message || ''}</span>
            </div>`;
        }).join('');

        if (state.logAutoScroll) {
            win.scrollTop = win.scrollHeight;
        }
    } catch (e) {
        console.error('Failed loading logs', e);
    }
}

// ─── Coverage Map ──────────────────────────────────────────────────────────
async function loadCoverageMap() {
    try {
        const data = await api('/api/coverage');
        const container = document.getElementById('coverage-matrix-container');
        if (!container) return;

        const cells = data.cells || {};
        const entries = Object.entries(cells);
        if (!entries.length) {
            container.innerHTML = '<div style="color: var(--text-muted); padding: 16px;">Coverage matrix empty.</div>';
            return;
        }

        container.innerHTML = entries.map(([key, count]) => `
            <div class="card" style="padding: 12px; border-left: 3px solid #3b82f6;">
                <div class="card-title" style="font-size: 11px;">${key}</div>
                <div class="card-value" style="font-size: 20px;">${count}</div>
                <div class="card-meta">Discovered Strategies</div>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed loading coverage map', e);
    }
}

// ─── Data and Integrity ────────────────────────────────────────────────────
async function loadDataIntegrity() {
    try {
        const data = await api('/api/data');

        // Split boundaries
        const boundEl = document.getElementById('data-boundaries-content');
        if (boundEl) {
            const b = data.boundaries || [];
            if (!b.length) {
                boundEl.innerHTML = '<div style="color: var(--text-muted);">Boundaries not frozen yet.</div>';
            } else {
                boundEl.innerHTML = b.map(item => `
                    <div style="font-family: var(--font-mono); font-size: 12px; margin-bottom: 8px;">
                        <strong>${item.timeframe || '1h'}:</strong> Train: ${item.train_start || ''} -> ${item.train_end || ''} | Val: ${item.validation_start || ''}
                    </div>
                `).join('');
            }
        }

        // Canary status
        const canEl = document.getElementById('data-canary-content');
        if (canEl) {
            const canary = data.canary || {};
            canEl.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="badge ${canary.passed ? 'badge-green' : 'badge-amber'}">${canary.passed ? 'VERIFIED' : 'PENDING'}</span>
                    <span style="font-size: 13px;">Last Check: ${canary.timestamp ? new Date(canary.timestamp * 1000).toLocaleString() : 'System startup'}</span>
                </div>
            `;
        }

        // Holdout access log
        const tbody = document.getElementById('holdout-audit-tbody');
        if (tbody) {
            const accesses = data.holdout_accesses || [];
            if (!accesses.length) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 24px;">No holdout evaluations requested.</td></tr>';
            } else {
                tbody.innerHTML = accesses.map(a => `
                    <tr>
                        <td class="mono">${a.timestamp ? new Date(a.timestamp * 1000).toISOString() : '—'}</td>
                        <td><strong>${a.strategy_id || '—'}</strong></td>
                        <td>${a.triggered_by || 'system'}</td>
                        <td><span class="badge ${a.result === 'passed' ? 'badge-green' : 'badge-red'}">${a.result || 'evaluated'}</span></td>
                    </tr>
                `).join('');
            }
        }
    } catch (e) {
        console.error('Failed loading data integrity', e);
    }
}

// ─── Settings ──────────────────────────────────────────────────────────────
async function loadSettings() {
    try {
        const data = await api('/api/settings');
        const container = document.getElementById('locked-thresholds-grid');
        if (container) {
            container.innerHTML = Object.entries(data.locked_gate_thresholds || {}).map(([k, v]) => `
                <div class="card" style="padding: 12px;">
                    <div class="card-title">${k.replace(/_/g, ' ')} <span class="badge badge-red">LOCKED</span></div>
                    <div class="card-value" style="font-size: 18px;">${v}</div>
                </div>
            `).join('');
        }

        const overrides = data.runtime_overrides || {};
        const tokEl = document.getElementById('setting-token-budget');
        if (tokEl && overrides.daily_token_budget) tokEl.value = overrides.daily_token_budget;
        const parEl = document.getElementById('setting-parallelism');
        if (parEl && overrides.max_parallel_backtests) parEl.value = overrides.max_parallel_backtests;
    } catch (e) {
        console.error('Failed loading settings', e);
    }
}

async function saveSettings() {
    try {
        const tokBudget = parseInt(document.getElementById('setting-token-budget')?.value, 10);
        const parallelism = parseInt(document.getElementById('setting-parallelism')?.value, 10);

        const payload = {};
        if (!isNaN(tokBudget) && tokBudget > 0) payload.daily_token_budget = tokBudget;
        if (!isNaN(parallelism) && parallelism > 0) payload.max_parallel_backtests = parallelism;

        await api('/api/settings', { method: 'PATCH', body: payload });
        alert('Runtime overrides updated successfully.');
    } catch (e) {
        alert('Failed saving settings: ' + e.message);
    }
}

// ─── Modals ────────────────────────────────────────────────────────────────
function showConfirmModal(title, message, onConfirm) {
    const modal = document.getElementById('confirm-modal');
    document.getElementById('confirm-modal-title').innerText = title;
    document.getElementById('confirm-modal-body').innerText = message;
    modal.classList.add('open');

    const btnOk = document.getElementById('confirm-modal-ok');
    btnOk.onclick = async () => {
        modal.classList.remove('open');
        await onConfirm();
    };
    document.getElementById('confirm-modal-cancel').onclick = () => {
        modal.classList.remove('open');
    };
}

function showLoginModal() {
    const token = prompt('Enter QuantForge Dashboard Token:');
    if (token) {
        api('/api/auth/login', { method: 'POST', body: { token } })
            .then(res => {
                state.csrfToken = res.csrf_token;
                window.location.reload();
            })
            .catch(err => alert('Authentication failed: ' + err.message));
    }
}

// ─── Init on DOM Load ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const savedTheme = localStorage.getItem('qf_theme');
        if (savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);
    } catch (e) {}

    setupNavigation();
    setupControls();

    // Check auth and initial status
    try {
        const st = await api('/api/engine/state');
        updateEngineUI(st);
        initSSE();
        loadFunnel();
        loadActivityOverview();
    } catch (e) {
        console.warn('Initial load unauthenticated or failed', e);
    }
});
