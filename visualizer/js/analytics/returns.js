// =====================================================================
    // SECTION 19 — ANALYTICS: RETURNS
    // =====================================================================
    function renderReturnsTab() {
      const el = document.getElementById('returnsTab');
      const snaps = getVisSnaps();
      if (snaps.length < 20) { el.innerHTML = '<div style="color:#555">Not enough data</div>'; return; }

      const mids = snaps.map(s => s.mid).filter(m => m != null && m > 0);
      const returns = [];
      for (let i = 1; i < mids.length; i++) returns.push(Math.log(mids[i] / mids[i - 1]));
      if (returns.length < 10) { el.innerHTML = '<div style="color:#555">Not enough returns</div>'; return; }

      const maxLag = Math.min(50, Math.floor(returns.length / 5));
      const mr = mean(returns), vr = vari(returns);
      const ac = [];
      for (let lag = 1; lag <= maxLag; lag++) {
        let s = 0, n = 0;
        for (let i = lag; i < returns.length; i++) { s += (returns[i] - mr) * (returns[i - lag] - mr); n++; }
        ac.push({ lag, v: n && vr ? +(s / (n * vr)).toFixed(6) : 0 });
      }

      const volW = Math.min(100, Math.floor(returns.length / 5));
      const rolVol = [];
      for (let i = volW; i < returns.length; i++) rolVol.push(std(returns.slice(i - volW, i)));

      const confBand = 1.96 / Math.sqrt(returns.length);
      const sigLags = ac.filter(a => Math.abs(a.v) > confBand);

      let h = `<div class="explain-box"><b>Autocorrelation of 1-tick log-returns.</b> <span class="good">Lag-1 negative → mean-reverting</span> (price snaps back — fade moves, place counter-trend orders). <span class="warn">Lag-1 positive → trending</span> (follow direction). Bars above/below the blue ± threshold line are <b>statistically significant</b> at 95% confidence. Return distribution: fat tails (high kurtosis &gt; 3) mean rare large moves.</div>`;
      h += `<div class="section-title">Return Stats (1-tick log)</div>`;
      h += `<div class="stat-grid">
    <div class="lbl">Mean</div><div class="val">${mean(returns).toExponential(2)}</div>
    <div class="lbl">Std Dev</div><div class="val">${std(returns).toExponential(2)}</div>
    <div class="lbl">Skewness</div><div class="val">${skew(returns).toFixed(3)}</div>
    <div class="lbl">Kurtosis</div><div class="val">${kurt(returns).toFixed(3)}</div>
    <div class="lbl">95% CI</div><div class="val">±${confBand.toFixed(4)}</div>
    <div class="lbl">Sig. AC lags</div><div class="val" style="color:${sigLags.length ? '#ff4488' : '#44ff88'}">${sigLags.length}</div>
  </div>`;
      if (sigLags.length) h += `<div style="font-size:9px;color:#ff4488;margin:3px 0">Mean-reversion signal: lags ${sigLags.map(a => a.lag).join(', ')}</div>`;

      h += `<div class="section-title">Autocorrelation (lags 1–${maxLag})</div>`;
      h += `<div class="mini-chart" style="height:140px"><canvas id="acC"></canvas></div>`;
      h += `<div class="section-title">Return Distribution</div>`;
      h += `<div class="mini-chart" style="height:90px"><canvas id="retHC"></canvas></div>`;
      h += `<div class="section-title">Rolling Volatility (window ${volW})</div>`;
      h += `<div class="mini-chart" style="height:80px"><canvas id="rvC"></canvas></div>`;
      el.innerHTML = h;

      setTimeout(() => {
        const c = document.getElementById('acC'); if (!c) return;
        const { ctx, w, h } = setupCanvas(c, c.parentElement);
        const maxAC = Math.max(confBand * 2, ...ac.map(a => Math.abs(a.v)));
        const bw = (w - 30) / ac.length;
        const cy = h / 2;
        // Conf band shading
        ctx.fillStyle = 'rgba(76,201,240,0.07)';
        const bh = (confBand / maxAC) * (h / 2 - 10);
        ctx.fillRect(20, cy - bh, w - 30, bh * 2);
        // Zero line
        ctx.strokeStyle = '#444'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(20, cy); ctx.lineTo(w, cy); ctx.stroke();
        // Conf lines
        ctx.strokeStyle = 'rgba(76,201,240,0.3)'; ctx.setLineDash([3, 4]);
        [{ y: cy - bh }, { y: cy + bh }].forEach(({ y }) => { ctx.beginPath(); ctx.moveTo(20, y); ctx.lineTo(w, y); ctx.stroke(); });
        ctx.setLineDash([]);
        // Bars
        for (let i = 0; i < ac.length; i++) {
          const v = ac[i].v;
          const barH = (Math.abs(v) / maxAC) * (h / 2 - 10);
          const x = 20 + i * bw;
          const sig = Math.abs(v) > confBand;
          ctx.fillStyle = sig ? (v > 0 ? '#44ff44' : '#ff4444') : '#444';
          if (v >= 0) ctx.fillRect(x + .5, cy - barH, bw - 1, barH);
          else ctx.fillRect(x + .5, cy, bw - 1, barH);
        }
        ctx.fillStyle = '#555'; ctx.font = '8px Courier New';
        ctx.fillText('1', 20, h - 2); ctx.textAlign = 'right'; ctx.fillText(maxLag, w, h - 2);
        ctx.fillText('+' + confBand.toFixed(3), w, cy - bh - 1);
        ctx.textAlign = 'start';
      }, 40);

      setTimeout(() => {
        const c = document.getElementById('retHC'); if (!c) return;
        const { ctx, w, h } = setupCanvas(c, c.parentElement);
        const hist = histo(returns, 40), maxC = Math.max(...hist.counts), bw = w / hist.counts.length;
        ctx.fillStyle = '#4cc9f0';
        for (let i = 0; i < hist.counts.length; i++) { const bh = (hist.counts[i] / maxC) * (h - 12); ctx.fillRect(i * bw + .5, h - bh, bw - 1, bh); }
        ctx.fillStyle = '#555'; ctx.font = '8px Courier New';
        ctx.fillText(hist.edges[0].toExponential(1), 2, h - 2);
        ctx.textAlign = 'right'; ctx.fillText(hist.edges[hist.edges.length - 1].toExponential(1), w - 2, h - 2); ctx.textAlign = 'start';
      }, 80);

      setTimeout(() => {
        const c = document.getElementById('rvC'); if (!c) return;
        const { ctx, w, h } = setupCanvas(c, c.parentElement);
        if (!rolVol.length) return;
        const mn = Math.min(...rolVol), mx = Math.max(...rolVol), rng = mx - mn || 1;
        ctx.strokeStyle = '#ff8800'; ctx.lineWidth = 1; ctx.beginPath();
        for (let i = 0; i < rolVol.length; i++) {
          const x = (i / (rolVol.length - 1)) * w, y = h - ((rolVol[i] - mn) / rng) * (h - 14);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.fillStyle = '#555'; ctx.font = '8px Courier New';
        ctx.fillText('σ=' + mn.toExponential(1), 2, h - 2);
        ctx.textAlign = 'right'; ctx.fillText(mx.toExponential(1), w - 2, 12); ctx.textAlign = 'start';
      }, 120);
    }
