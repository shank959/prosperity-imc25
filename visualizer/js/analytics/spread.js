// =====================================================================
    // SECTION 18 — ANALYTICS: SPREAD
    // =====================================================================
    function renderSpreadTab() {
      const el = document.getElementById('spreadTab');
      const snaps = getVisSnaps();
      if (snaps.length < 5) { el.innerHTML = '<div style="color:#555">No data</div>'; return; }

      const spreads = snaps.filter(s => s.bids.length && s.asks.length)
        .map(s => ({ ts: s.ts, v: Math.min(...s.asks.map(a => a.price)) - Math.max(...s.bids.map(b => b.price)) }));
      const vals = spreads.map(s => s.v);

      let h = `<div class="explain-box"><b>Spread = best ask − best bid.</b> Wide spread = more edge per fill (good for market-making). Tight spread = competitive market, harder to profit. <span class="good">P90 is a safe passive quote width</span> — quote inside P90 for fills. <span class="warn">Mean &lt; 2 → very tight, be cautious.</span> Time series shows when spread opens/closes.</div>`;
      h += `<div class="section-title">Bid-Ask Spread</div>`;
      h += `<div class="stat-grid">
    <div class="lbl">Mean</div><div class="val">${mean(vals).toFixed(2)}</div>
    <div class="lbl">Median</div><div class="val">${med(vals).toFixed(2)}</div>
    <div class="lbl">Std Dev</div><div class="val">${std(vals).toFixed(2)}</div>
    <div class="lbl">Min</div><div class="val">${Math.min(...vals).toFixed(0)}</div>
    <div class="lbl">Max</div><div class="val">${Math.max(...vals).toFixed(0)}</div>
    <div class="lbl">P10</div><div class="val">${pctile(vals, 10).toFixed(2)}</div>
    <div class="lbl">P90</div><div class="val">${pctile(vals, 90).toFixed(2)}</div>
    <div class="lbl">N</div><div class="val">${vals.length}</div>
  </div>`;
      h += `<div class="section-title">Time Series</div><div class="mini-chart" style="height:100px"><canvas id="sTsC"></canvas></div>`;
      h += `<div class="section-title">Distribution</div><div class="mini-chart" style="height:90px"><canvas id="sHC"></canvas></div>`;
      el.innerHTML = h;

      setTimeout(() => {
        const c = document.getElementById('sTsC'); if (!c) return;
        const { ctx, w, h } = setupCanvas(c, c.parentElement);
        const ds = downsample(spreads, 400);
        const mn = Math.min(...vals), mx = Math.max(...vals), rng = mx - mn || 1;
        drawGrid(ctx, w, h, ds[0].ts, ds[ds.length - 1].ts, mn - rng * .1, mx + rng * .1);
        ctx.strokeStyle = '#4cc9f0'; ctx.lineWidth = 1; ctx.beginPath(); let st = false;
        for (const s of ds) { const x = ((s.ts - ds[0].ts) / (ds[ds.length - 1].ts - ds[0].ts || 1)) * w; const y = h - ((s.v - mn + rng * .1) / (rng * 1.2)) * h; if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y); }
        ctx.stroke();
      }, 40);
      setTimeout(() => {
        const c = document.getElementById('sHC'); if (!c) return;
        const { ctx, w, h } = setupCanvas(c, c.parentElement);
        const hist = histo(vals, 25); const maxC = Math.max(...hist.counts); const bw = w / hist.counts.length;
        ctx.fillStyle = '#4cc9f0';
        for (let i = 0; i < hist.counts.length; i++) { const bh = (hist.counts[i] / maxC) * (h - 14); ctx.fillRect(i * bw + .5, h - bh, bw - 1, bh); }
        ctx.fillStyle = '#555'; ctx.font = '8px Courier New';
        ctx.fillText(hist.edges[0].toFixed(1), 2, h - 2); ctx.textAlign = 'right';
        ctx.fillText(hist.edges[hist.edges.length - 1].toFixed(1), w - 2, h - 2); ctx.textAlign = 'start';
      }, 80);
    }
