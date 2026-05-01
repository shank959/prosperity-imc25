// =====================================================================
    // SECTION 13 — POSITION / PnL PANEL
    // =====================================================================
    function renderPosPnl() {
      if (!ppCtx) return;
      const ctx = ppCtx, w = ppW, h = ppH;
      ctx.clearRect(0, 0, w, h);
      if (!currentProduct) return;

      const snaps = getRange();
      const ds = downsample(snaps, maxPoints);
      if (!ds.length) return;

      const posLimit = PRODUCT_CONFIG[currentProduct]?.posLimit || 80;
      const posYMin = -posLimit - 8, posYMax = posLimit + 8;
      // Prefer trade-based MTM PnL; fall back to CSV profit_and_loss column.
      const pnls = ds.map(s => s.mtmPnl ?? s.pnl).filter(p => p != null);
      const pnlMin = pnls.length ? Math.min(...pnls) - 100 : 0;
      const pnlMax = pnls.length ? Math.max(...pnls) + 100 : 1000;

      const posToY = p => h - ((p - posYMin) / (posYMax - posYMin)) * h;
      const pnlToY = p => h - ((p - pnlMin) / (pnlMax - pnlMin)) * h;

      // Limit warning bands
      ctx.fillStyle = COLORS.posLimit;
      ctx.fillRect(0, 0, w, posToY(posLimit * 0.75));
      ctx.fillRect(0, posToY(-posLimit * 0.75), w, h);

      // X grid (aligned with main chart)
      ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 0.5;
      const xi = niceInterval(vp.xMax - vp.xMin, 9);
      for (let v = Math.ceil(vp.xMin / xi) * xi; v <= vp.xMax; v += xi) {
        const x = ppTsToX(v);
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }

      // Zero line
      ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
      const y0 = posToY(0);
      ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(w, y0); ctx.stroke();

      // Position limits dashed
      ctx.strokeStyle = 'rgba(255,60,60,0.4)'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
      [posLimit, -posLimit].forEach(lim => {
        const y = posToY(lim);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      });
      ctx.setLineDash([]);

      // PnL line (right axis, green) — MTM from trades + FV for remaining position.
      // Uses snap.mtmPnl when available (computed from own fills + FV); falls back to CSV column.
      if (pnls.length) {
        ctx.strokeStyle = COLORS.pnl; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.7;
        ctx.beginPath(); let st = false;
        for (const s of ds) {
          const pnlVal = s.mtmPnl ?? s.pnl;
          if (pnlVal == null) continue;
          const x = ppTsToX(s.ts), y = pnlToY(pnlVal);
          if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
        const hasMtm = ds.some(s => s.mtmPnl != null);
        ctx.fillStyle = COLORS.pnl; ctx.font = '9px Courier New'; ctx.textAlign = 'right';
        ctx.fillText('PnL=' + pnls[pnls.length - 1].toFixed(0) + (hasMtm ? ' (MTM+FV)' : ' (CSV)'), w - 4, 12);
        ctx.textAlign = 'start';
      }

      // Total portfolio PnL from graphLog (official IMC JSON only)
      if (Store.graphLog && Store.graphLog.length > 1) {
        const visGL = Store.graphLog.filter(g => g.ts >= vp.xMin && g.ts <= vp.xMax);
        if (visGL.length > 1) {
          const glVals = visGL.map(g => g.value);
          const glMin = Math.min(...glVals), glMax = Math.max(...glVals);
          const glRange = glMax - glMin || 1;
          const glToY = v => h - ((v - (glMin - glRange * 0.05)) / (glRange * 1.1)) * h;

          ctx.strokeStyle = '#00ddff'; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.8;
          ctx.beginPath(); let st = false;
          for (const g of visGL) {
            const x = ppTsToX(g.ts), y = glToY(g.value);
            if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y);
          }
          ctx.stroke(); ctx.globalAlpha = 1;

          ctx.fillStyle = '#00ddff'; ctx.font = '9px Courier New'; ctx.textAlign = 'right';
          ctx.fillText('Total=' + glVals[glVals.length - 1].toFixed(0), w - 4, 24);
          ctx.textAlign = 'start';
        }
      }

      // Sum-all-products PnL line (when no graphLog and multiple products loaded)
      if ((!Store.graphLog || !Store.graphLog.length) && Store.products.length > 1) {
        const allPnlMap = new Map();
        for (const prod of Store.products) {
          for (const s of (Store.data[prod] || [])) {
            if (s.pnl == null || s.ts < vp.xMin || s.ts > vp.xMax) continue;
            allPnlMap.set(s.ts, (allPnlMap.get(s.ts) || 0) + s.pnl);
          }
        }
        if (allPnlMap.size > 1) {
          const entries = [...allPnlMap.entries()].sort((a, b) => a[0] - b[0]);
          const sumVals = entries.map(e => e[1]);
          const sumMin = Math.min(...sumVals), sumMax = Math.max(...sumVals);
          const sumRange = sumMax - sumMin || 1;
          const sumToY = v => h - ((v - (sumMin - sumRange * 0.05)) / (sumRange * 1.1)) * h;

          ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.6;
          ctx.setLineDash([4, 3]);
          ctx.beginPath(); let st = false;
          for (const [ts, val] of entries) {
            const x = ppTsToX(ts), y = sumToY(val);
            if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y);
          }
          ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;

          ctx.fillStyle = '#ffffff'; ctx.font = '9px Courier New'; ctx.textAlign = 'right';
          ctx.fillText('Sum=' + sumVals[sumVals.length - 1].toFixed(0), w - 4, 24);
          ctx.textAlign = 'start';
        }
      }

      // Position line (left axis, yellow)
      const positions = ds.map(s => s.pos).filter(p => p != null);
      if (positions.length) {
        ctx.strokeStyle = COLORS.position; ctx.lineWidth = 1.5;
        ctx.beginPath(); let st = false;
        for (const s of ds) {
          if (s.pos == null) continue;
          const x = ppTsToX(s.ts), y = posToY(s.pos);
          if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.fillStyle = COLORS.position; ctx.font = '9px Courier New';
        ctx.fillText('Pos=' + positions[positions.length - 1], 4, 12);
      }

      // Y labels
      ctx.fillStyle = '#3a4a5a'; ctx.font = '8px Courier New';
      ctx.fillText('+' + posLimit, 4, posToY(posLimit) + 9);
      ctx.fillText('-' + posLimit, 4, posToY(-posLimit) - 2);
    }
