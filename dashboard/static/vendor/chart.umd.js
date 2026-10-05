/**
 * Vendored Zero-Dependency Offline Canvas Chart Library (Chart.js UMD compatible subset).
 * Supports line charts, bar charts, multi-dataset equity curves, and drawdowns.
 */
(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? module.exports = factory() :
    typeof define === 'function' && define.amd ? define(factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, global.Chart = factory());
})(this, function () {
    'use strict';

    class Chart {
        constructor(ctx, config) {
            this.ctx = ctx.getContext ? ctx.getContext('2d') : ctx;
            this.canvas = this.ctx.canvas;
            this.config = config || {};
            this.type = this.config.type || 'line';
            this.data = this.config.data || { labels: [], datasets: [] };
            this.options = this.config.options || {};

            this.width = this.canvas.width;
            this.height = this.canvas.height;
            this.render();
        }

        destroy() {
            this.ctx.clearRect(0, 0, this.width, this.height);
        }

        update() {
            this.render();
        }

        render() {
            const ctx = this.ctx;
            const w = this.canvas.clientWidth || this.canvas.width;
            const h = this.canvas.clientHeight || this.canvas.height;
            this.canvas.width = w;
            this.canvas.height = h;

            ctx.clearRect(0, 0, w, h);

            const datasets = this.data.datasets || [];
            if (!datasets.length) return;

            const padding = { top: 20, right: 30, bottom: 40, left: 60 };
            const plotW = w - padding.left - padding.right;
            const plotH = h - padding.top - padding.bottom;

            // Compute global min / max
            let allVals = [];
            datasets.forEach(ds => {
                if (Array.isArray(ds.data)) allVals.push(...ds.data);
            });
            if (!allVals.length) return;

            let minVal = Math.min(...allVals);
            let maxVal = Math.max(...allVals);
            if (minVal === maxVal) { minVal -= 1; maxVal += 1; }
            if (this.type === 'bar' && minVal > 0) minVal = 0;

            const valRange = maxVal - minVal;

            // Draw Gridlines & Y-Axis Labels
            ctx.strokeStyle = '#2d3748';
            ctx.fillStyle = '#a0aec0';
            ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
            ctx.textAlign = 'right';

            const numTicks = 5;
            for (let i = 0; i <= numTicks; i++) {
                const tickVal = minVal + (valRange * (numTicks - i)) / numTicks;
                const y = padding.top + (plotH * i) / numTicks;

                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(w - padding.right, y);
                ctx.stroke();

                ctx.fillText(tickVal.toLocaleString(undefined, { maximumFractionDigits: 2 }), padding.left - 8, y + 4);
            }

            // Draw Datasets
            const labels = this.data.labels || [];
            const numPoints = Math.max(...datasets.map(d => d.data.length), labels.length);

            datasets.forEach(ds => {
                const data = ds.data || [];
                const color = ds.borderColor || '#3b82f6';
                const bgColor = ds.backgroundColor || 'rgba(59, 130, 246, 0.2)';

                if (this.type === 'line') {
                    ctx.beginPath();
                    ctx.strokeStyle = color;
                    ctx.lineWidth = ds.borderWidth || 2;

                    data.forEach((val, idx) => {
                        const x = padding.left + (plotW * idx) / Math.max(1, data.length - 1);
                        const y = padding.top + plotH - ((val - minVal) / valRange) * plotH;
                        if (idx === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    });
                    ctx.stroke();

                    // Optional area fill
                    if (ds.fill) {
                        const firstX = padding.left;
                        const lastX = padding.left + plotW;
                        const bottomY = padding.top + plotH;
                        ctx.lineTo(lastX, bottomY);
                        ctx.lineTo(firstX, bottomY);
                        ctx.closePath();
                        ctx.fillStyle = bgColor;
                        ctx.fill();
                    }
                } else if (this.type === 'bar') {
                    const barCount = data.length;
                    const barWidth = Math.max(4, (plotW / barCount) * 0.7);
                    const gap = (plotW / barCount) * 0.3;

                    ctx.fillStyle = color;
                    data.forEach((val, idx) => {
                        const x = padding.left + idx * (barWidth + gap) + gap / 2;
                        const barHeight = ((val - minVal) / valRange) * plotH;
                        const y = padding.top + plotH - barHeight;
                        ctx.fillRect(x, y, barWidth, barHeight);
                    });
                }
            });

            // Draw X-Axis Labels (sample every Nth label)
            ctx.textAlign = 'center';
            ctx.fillStyle = '#a0aec0';
            const labelStep = Math.max(1, Math.ceil(labels.length / 8));
            labels.forEach((lbl, idx) => {
                if (idx % labelStep === 0) {
                    const x = padding.left + (plotW * idx) / Math.max(1, labels.length - 1);
                    ctx.fillText(String(lbl).substring(0, 10), x, h - padding.bottom + 18);
                }
            });
        }
    }

    return Chart;
});
