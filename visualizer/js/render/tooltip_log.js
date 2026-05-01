// =====================================================================
    // SECTION 14 — HOVER / TOOLTIP / LOG VIEWER
    // =====================================================================
    function showTooltip(cx, cy, lx) {
      const tooltip = document.getElementById('tooltip');
      const snaps = getVisSnaps();
      if (!snaps.length || !currentProduct) { tooltip.style.display = 'none'; return; }

      const ts = xToTs(lx);
      const snap = findNearest(snaps, ts);
      if (!snap) { tooltip.style.display = 'none'; return; }

      const spread = snap.bids.length && snap.asks.length
        ? (Math.min(...snap.asks.map(a => a.price)) - Math.max(...snap.bids.map(b => b.price))).toFixed(0)
        : '—';

      // Full OB table
      let html = `<b style="color:#4cc9f0">T=${snap.ts}</b>  Day ${snap.day}`;
      if (snap.fair != null) html += `  <span style="color:#44ff88">FV=${snap.fair.toFixed(1)}</span>`;
      if (snap.computedFV != null) html += `  <span style="color:#88ff88">BotFV=${snap.computedFV}</span>`;
      html += `  Sprd=<b>${spread}</b>`;
      if (snap.remainingCapacity != null) html += `  <span style="color:#aaa">Cap=${snap.remainingCapacity}</span>`;
      if (snap.obDepthSummary) {
        const ob = snap.obDepthSummary;
        html += `  <span style="color:#666">${ob.levels}lvl B${ob.bidVol}/A${ob.askVol}</span>`;
      }
      html += `<br>`;

      // Orderbook: 3 levels side by side
      html += `<table style="border-collapse:collapse;width:100%;margin:3px 0;font-size:9px">`;
      html += `<tr><td style="color:#4488ff;padding:0 2px">BID</td><td style="color:#4488ff">VOL</td><td style="width:6px"></td><td style="color:#ff4444">ASK</td><td style="color:#ff4444">VOL</td></tr>`;
      const maxLvl = Math.max(snap.bids.length, snap.asks.length, 3);
      for (let i = 0; i < maxLvl; i++) {
        const b = snap.bids[i], a = snap.asks[i];
        const bStr = b ? `<td style="color:#88aaff;padding:0 2px">${b.price}</td><td style="color:#88aaff">${b.vol}</td>` : '<td colspan="2" style="color:#333">—</td>';
        const aStr = a ? `<td style="color:#ff8888;padding:0 2px">${a.price}</td><td style="color:#ff8888">${a.vol}</td>` : '<td colspan="2" style="color:#333">—</td>';
        const lbl = i === 0 ? 'L1' : i === 1 ? 'L2' : 'L3';
        html += `<tr>${bStr}<td style="color:#444;font-size:8px;text-align:center">${lbl}</td>${aStr}</tr>`;
      }
      html += `</table>`;

      // Position/PnL
      if (snap.pos != null) {
        const posColor = Math.abs(snap.pos) > 60 ? '#ff4444' : Math.abs(snap.pos) > 40 ? '#ffaa00' : '#aaa';
        html += `<div>Pos: <b style="color:${posColor}">${snap.pos}</b>`;
        const pnlVal = snap.mtmPnl ?? snap.pnl;
        if (pnlVal != null) {
          const pnlLabel = snap.mtmPnl != null ? 'MTM+FV' : 'CSV';
          html += `  PnL: <b style="color:${pnlVal >= 0 ? '#44ff88' : '#ff4444'}">${pnlVal.toFixed(0)}</b> <span style="color:#555;font-size:8px">(${pnlLabel})</span>`;
        }
        html += `</div>`;
      }

      // Bot orders — split Make vs Clear
      if (snap.botOrders && snap.botOrders.length) {
        const makeOrders = snap.botOrders.filter(o => o.type !== 'clear');
        const clearOrders = snap.botOrders.filter(o => o.type === 'clear');
        if (makeOrders.length) {
          const bids2 = makeOrders.filter(o => o.qty > 0);
          const asks2 = makeOrders.filter(o => o.qty < 0);
          html += `<div style="margin-top:3px"><b style="color:#fff">★ Make:</b> `;
          if (bids2.length) html += bids2.map(o => `<span class="bid-c">BID ${o.qty}@${o.price}</span>`).join(' ');
          if (asks2.length) html += asks2.map(o => `<span class="ask-c"> ASK ${Math.abs(o.qty)}@${o.price}</span>`).join(' ');
          html += `</div>`;
        }
        if (clearOrders.length) {
          const bids2 = clearOrders.filter(o => o.qty > 0);
          const asks2 = clearOrders.filter(o => o.qty < 0);
          html += `<div style="margin-top:2px"><b style="color:#4cc9f0">◇ Clear:</b> `;
          if (bids2.length) html += bids2.map(o => `<span class="bid-c">BID ${o.qty}@${o.price}</span>`).join(' ');
          if (asks2.length) html += asks2.map(o => `<span class="ask-c"> ASK ${Math.abs(o.qty)}@${o.price}</span>`).join(' ');
          html += `</div>`;
        }
      }

      // Trades at/near this tick
      const trades = Store.trades.filter(t => t.symbol === currentProduct && Math.abs(t.ts - snap.ts) <= 200);
      if (trades.length) {
        html += `<div style="margin-top:3px"><b>Trades (±200ms):</b><br>`;
        for (const t of trades.slice(0, 8)) {
          const clz = classifyTrade(t, snap);
          const c = TRADE_COLORS[clz];
          const isBuy = t.buyer === 'SUBMISSION' || (t.buyer && !t.seller);
          const side = t.buyer === 'SUBMISSION' ? '▲BUY' : t.seller === 'SUBMISSION' ? '▼SELL' : isBuy ? '▲' : '▼';
          const edge = snap.fair != null ? (t.price - snap.fair > 0 ? '+' : '') + (t.price - snap.fair).toFixed(1) : '';
          html += `<span style="color:${c}">[${clz}] ${side} ${t.qty}@${t.price} <i>${edge}</i></span><br>`;
        }
        html += `</div>`;
      }

      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      const tw = 260, th = 260;
      let tx = cx + 16, ty = cy - 8;
      if (tx + tw > window.innerWidth) tx = cx - tw - 14;
      if (ty + th > window.innerHeight) ty = window.innerHeight - th - 10;
      if (ty < 5) ty = 5;
      tooltip.style.left = tx + 'px';
      tooltip.style.top = ty + 'px';

      updateLogViewer(snap);
    }

    function updateLogViewer(snap) {
      document.getElementById('logTs').textContent = snap ? snap.ts : '—';
      const el = document.getElementById('logContent');
      if (!snap || !snap.logLines || !snap.logLines.length) {
        el.innerHTML = '<span style="color:#333">No log data at this timestamp</span>';
        return;
      }
      el.innerHTML = snap.logLines.map(line => {
        let cls = '';
        if (/^ORD\|.*\|MAKE\|/i.test(line)) cls = 'log-make';
        else if (/^ORD\|.*\|TAKE\|/i.test(line)) cls = 'log-take';
        else if (/^ORD\|.*\|CLEAR\|/i.test(line)) cls = 'log-clear';
        else if (/^MT\|/i.test(line)) cls = 'log-mt';
        else if (/MAKE OFFER|MAKE BID/i.test(line)) cls = 'log-make';
        else if (/TAKE/i.test(line)) cls = 'log-take';
        else if (/CLEAR/i.test(line)) cls = 'log-clear';
        return `<div class="${cls}">${escHtml(line)}</div>`;
      }).join('');
    }

    function escHtml(s) {
      return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // =====================================================================
    // SECTION 15 — MAIN CHART INTERACTION
    // =====================================================================
    (function () {
      const canvas = document.getElementById('mainCanvas');

      canvas.addEventListener('mousedown', e => {
        const r = canvas.getBoundingClientRect();
        const x = e.clientX - r.left;
        if (e.shiftKey || e.button === 1) {
          // Pan: X only (Y auto-fits on every render)
          dragState = { type: 'pan', sx: x, x0: vp.xMin, x1: vp.xMax };
        } else {
          dragState = { type: 'zoom', sx: x, cx: x };
        }
      });

      canvas.addEventListener('mousemove', e => {
        const r = canvas.getBoundingClientRect();
        const x = e.clientX - r.left;
        hoverTs = xToTs(x); // update crosshair immediately
        if (dragState) {
          if (dragState.type === 'zoom') {
            dragState.cx = x; scheduleRender();
          } else {
            // Pan X only
            const dx = xToTs(dragState.sx) - xToTs(x);
            vp.xMin = dragState.x0 + dx; vp.xMax = dragState.x1 + dx;
            dragState.sx = x; dragState.x0 = vp.xMin; dragState.x1 = vp.xMax;
            scheduleRender();
          }
          return;
        }
        scheduleRender(); // redraw crosshair
        clearTimeout(tooltipTimer);
        tooltipTimer = setTimeout(() => showTooltip(e.clientX, e.clientY, x), 25);
      });

      canvas.addEventListener('mouseup', e => {
        if (!dragState) return;
        if (dragState.type === 'zoom') {
          const r = canvas.getBoundingClientRect();
          const x = e.clientX - r.left;
          const x1 = Math.min(dragState.sx, x), x2 = Math.max(dragState.sx, x);
          if (x2 - x1 > 8) {
            // Zoom X only; Y will auto-fit via renderMain
            vp.xMin = xToTs(x1); vp.xMax = xToTs(x2);
          }
        }
        dragState = null; scheduleRender();
      });

      canvas.addEventListener('mouseleave', () => {
        dragState = null; hoverTs = null;
        clearTimeout(tooltipTimer);   // cancel any pending show-tooltip call
        document.getElementById('tooltip').style.display = 'none';
        scheduleRender();
      });

      canvas.addEventListener('wheel', e => {
        e.preventDefault();
        const r = canvas.getBoundingClientRect();
        const mx = e.clientX - r.left;
        // Zoom X only — Y auto-fits on render
        const factor = e.deltaY > 0 ? 1.18 : 1 / 1.18;
        const mts = xToTs(mx);
        vp.xMin = mts - (mts - vp.xMin) * factor;
        vp.xMax = mts + (vp.xMax - mts) * factor;
        scheduleRender();
      }, { passive: false });

      canvas.addEventListener('dblclick', () => { autoFitViewport(); scheduleRender(); });
    })();

    // Also wheel on pos/pnl for X zoom
    document.getElementById('posPnlCanvas').addEventListener('wheel', e => {
      e.preventDefault();
      const r = document.getElementById('posPnlCanvas').getBoundingClientRect();
      const mx = e.clientX - r.left;
      const factor = e.deltaY > 0 ? 1.18 : 1 / 1.18;
      const mts = vp.xMin + (mx / ppW) * (vp.xMax - vp.xMin);
      vp.xMin = mts - (mts - vp.xMin) * factor;
      vp.xMax = mts + (vp.xMax - mts) * factor;
      scheduleRender();
    }, { passive: false });
