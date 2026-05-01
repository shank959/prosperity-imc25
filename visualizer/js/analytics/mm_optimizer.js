// =====================================================================
    // SECTION 21 — ANALYTICS: MM OPTIMIZER
    // =====================================================================
    function renderMMTab() {
      const el = document.getElementById('mmTab');
      const snaps = getVisSnaps();
      if (snaps.length < 50) { el.innerHTML = '<div style="color:#555">Need more data</div>'; return; }
      el.innerHTML = `<div style="color:#4cc9f0;padding:8px">Running grid search (${snaps.length} ticks)...</div>`;

      setTimeout(() => {
        const TWs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        const MWs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        const posLimit = PRODUCT_CONFIG[currentProduct]?.posLimit || 80;
        const grid = []; let minP = Infinity, maxP = -Infinity;
        let bestTW = 1, bestMW = 2, bestPnl = -Infinity;

        for (const tw of TWs) {
          const row = [];
          for (const mw of MWs) {
            const pnl = simulateMM(snaps, tw, mw, posLimit);
            row.push(pnl);
            if (pnl < minP) minP = pnl; if (pnl > maxP) maxP = pnl;
            if (pnl > bestPnl) { bestPnl = pnl; bestTW = tw; bestMW = mw; }
          }
          grid.push(row);
        }

        let h = `<div class="explain-box"><b>Grid search over market-making parameters.</b> <b>take_width</b>: how aggressively to lift/hit — <span class="good">lower = more fills but higher adverse selection risk</span>. <b>make_width</b>: how wide to quote passively — <span class="good">wider = more edge per fill, fewer fills</span>. Best cell = highest simulated PnL. <span class="warn">Use as directional guidance, not exact targets — simulator ignores partial fills and queue priority.</span></div>`;
        h += `<div class="section-title">MM Parameter Grid Search</div>`;
        h += `<div class="stat-grid">
      <div class="lbl">Best take_w</div><div class="val">${bestTW}</div>
      <div class="lbl">Best make_w</div><div class="val">${bestMW}</div>
      <div class="lbl">Est. PnL</div><div class="val" style="color:${bestPnl > 0 ? '#44ff44' : '#ff4444'}">${bestPnl.toFixed(0)}</div>
      <div class="lbl">Current take_w</div><div class="val">1</div>
      <div class="lbl">Current make_w</div><div class="val">2</div>
    </div>`;
        h += `<div class="heatmap-legend"><span>${minP.toFixed(0)}</span><div class="bar"></div><span>${maxP.toFixed(0)}</span></div>`;
        h += `<div class="mini-chart" style="height:230px"><canvas id="hmC"></canvas></div>`;
        h += `<div style="font-size:8px;color:#555;margin-top:3px">Y=take_w (rows) X=make_w (cols). White border=best.</div>`;
        el.innerHTML = h;

        setTimeout(() => {
          const c = document.getElementById('hmC'); if (!c) return;
          const { ctx, w, h } = setupCanvas(c, c.parentElement);
          const rows = grid.length, cols = grid[0].length;
          const cellW = (w - 28) / cols, cellH = (h - 20) / rows;
          const rng = maxP - minP || 1;
          for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
              const v = grid[i][j], norm = (v - minP) / rng;
              let r, g, b;
              if (norm < 0.5) { r = 255; g = Math.floor(norm * 2 * 200); b = 50; }
              else { r = Math.floor((1 - norm) * 2 * 255); g = 220; b = 50; }
              ctx.fillStyle = `rgb(${r},${g},${b})`;
              ctx.fillRect(28 + j * cellW, i * cellH, cellW - 1, cellH - 1);
              ctx.fillStyle = norm > .2 && norm < .8 ? '#000' : '#fff';
              ctx.font = '8px Courier New'; ctx.textAlign = 'center';
              ctx.fillText(v.toFixed(0), 28 + j * cellW + cellW / 2, i * cellH + cellH / 2 + 3);
            }
          }
          // Highlight best
          const bi = TWs.indexOf(bestTW), bj = MWs.indexOf(bestMW);
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
          ctx.strokeRect(28 + bj * cellW, bi * cellH, cellW - 1, cellH - 1);
          // Axis labels
          ctx.fillStyle = '#666'; ctx.font = '8px Courier New';
          ctx.textAlign = 'center';
          for (let j = 0; j < cols; j++) ctx.fillText(MWs[j], 28 + j * cellW + cellW / 2, rows * cellH + 11);
          ctx.textAlign = 'right';
          for (let i = 0; i < rows; i++) ctx.fillText(TWs[i], 24, i * cellH + cellH / 2 + 3);
          ctx.textAlign = 'start';
        }, 30);
      }, 20);
    }

    function simulateMM(snaps, takeW, makeW, posLimit) {
      let pos = 0, pnl = 0;
      for (const s of snaps) {
        if (!s.fair) continue;
        const fv = s.fair;
        for (const ask of s.asks) if (ask.price <= fv - takeW && pos < posLimit) { const q = Math.min(ask.vol, posLimit - pos); pos += q; pnl -= ask.price * q; }
        for (const bid of s.bids) if (bid.price >= fv + takeW && pos > -posLimit) { const q = Math.min(bid.vol, posLimit + pos); pos -= q; pnl += bid.price * q; }
        if (Math.abs(pos) > posLimit * 0.6) { const clr = Math.floor(Math.abs(pos) * 0.08); if (pos > 0) { pos -= clr; pnl += fv * clr; } else { pos += clr; pnl -= fv * clr; } }
      }
      const lf = snaps[snaps.length - 1]?.fair || 0;
      pnl += pos * lf;
      return pnl;
    }
