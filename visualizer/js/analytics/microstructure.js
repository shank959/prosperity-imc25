// =====================================================================
    // SECTION 20 — ANALYTICS: MARKET MICROSTRUCTURE
    // =====================================================================
    function renderBotsTab() {
      const el = document.getElementById('botsTab');
      const snaps = getVisSnaps();
      const trades = Store.trades.filter(t => t.symbol === currentProduct);
      if (!snaps.length) { el.innerHTML = '<div style="color:#555">No data</div>'; return; }

      const mktTrades = trades.filter(t => t.buyer !== 'SUBMISSION' && t.seller !== 'SUBMISSION');
      const ownTrades = trades.filter(t => t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION');

      // --- SECTION A: Fair value stability (WallMid changes) ---
      // How often does fair value move? Rapid changes = liquid market hard to quote
      const fairVals = snaps.map(s => s.fair).filter(f => f != null);
      const fairChanges = [];
      for (let i = 1; i < fairVals.length; i++) if (fairVals[i] !== fairVals[i - 1]) fairChanges.push(Math.abs(fairVals[i] - fairVals[i - 1]));
      const fairChangePct = fairVals.length ? (fairChanges.length / fairVals.length * 100).toFixed(1) : 0;

      // --- SECTION B: Where does volume trade relative to fair value? ---
      const distFromFair = mktTrades.map(t => {
        const s = findNearest(snaps, t.ts);
        return s && s.fair != null ? t.price - s.fair : null;
      }).filter(f => f != null);
      const buyDist = distFromFair.filter(d => d < 0); // bought below fair (aggressive buys under mid = unlikely)
      const sellDist = distFromFair.filter(d => d > 0);

      // --- SECTION C: Spread quality (how much of spread is "real"?) ---
      const spreads = snaps.filter(s => s.bids.length && s.asks.length)
        .map(s => Math.min(...s.asks.map(a => a.price)) - Math.max(...s.bids.map(b => b.price)));
      const tightPct = spreads.length ? (spreads.filter(s => s <= 2).length / spreads.length * 100).toFixed(0) : 0;

      // --- SECTION D: Bot quote analysis ---
      const snapsWithBotOrders = snaps.filter(s => s.botOrders && s.botOrders.length > 0);
      const botMakeOrders = snapsWithBotOrders.flatMap(s => s.botOrders.filter(o => o.type !== 'clear'));
      const botBidLevels = snapsWithBotOrders.flatMap(s => s.botOrders.filter(o => o.type !== 'clear' && o.qty > 0).map(o => o.price - (s.fair || 0)));
      const botAskLevels = snapsWithBotOrders.flatMap(s => s.botOrders.filter(o => o.type !== 'clear' && o.qty < 0).map(o => o.price - (s.fair || 0)));
      const snapsWithClearOrders = snaps.filter(s => s.botOrders && s.botOrders.some(o => o.type === 'clear'));
      const clearBidLevels = snapsWithClearOrders.flatMap(s => s.botOrders.filter(o => o.type === 'clear' && o.qty > 0).map(o => o.price - (s.fair || 0)));
      const clearAskLevels = snapsWithClearOrders.flatMap(s => s.botOrders.filter(o => o.type === 'clear' && o.qty < 0).map(o => o.price - (s.fair || 0)));

      // --- SECTION E: Price impact of large trades ---
      const bigTrades = mktTrades.filter(t => t.qty >= 5);
      const impacts = [];
      for (const t of bigTrades) {
        const idx = snaps.findIndex(s => s.ts >= t.ts);
        if (idx < 0 || idx >= snaps.length - 3) continue;
        const before = snaps[idx].fair, after = snaps[Math.min(idx + 3, snaps.length - 1)].fair;
        if (before != null && after != null) {
          const wasBuy = t.buyer && t.buyer !== 'SUBMISSION';
          impacts.push(wasBuy ? after - before : before - after);
        }
      }

      // --- SECTION F: Unique qty signatures ---
      const qtyFreq = {};
      for (const t of mktTrades) { qtyFreq[t.qty] = (qtyFreq[t.qty] || 0) + 1; }
      const topQtys = Object.entries(qtyFreq).sort((a, b) => b[1] - a[1]).slice(0, 10);

      // Daily ranges
      const dailyStats = {};
      for (const s of snaps) {
        if (s.mid == null) continue;
        if (!dailyStats[s.day]) dailyStats[s.day] = { min: Infinity, max: -Infinity, count: 0 };
        if (s.mid < dailyStats[s.day].min) dailyStats[s.day].min = s.mid;
        if (s.mid > dailyStats[s.day].max) dailyStats[s.day].max = s.mid;
        dailyStats[s.day].count++;
      }

      let h = `<div class="explain-box"><b>Market microstructure analysis.</b> <b>Qty signatures</b>: bots typically trade in fixed sizes (2–5 units at exact prices) — recurring sizes = bot fingerprint. <b>Price impact</b>: FV change 3 ticks after a big trade — <span class="bad">positive = adverse selection risk</span>. <b>Daily range</b>: overall volatility. <b>Extreme trades</b>: activity near daily high/low often marks bot reversion points or informed flow.</div>`;
      h += `<div class="section-title">Market Overview</div>`;
      h += `<div class="stat-grid">
    <div class="lbl">Mkt trades</div><div class="val">${mktTrades.length}</div>
    <div class="lbl">Own fills (F)</div><div class="val">${ownTrades.length}</div>
    <div class="lbl">Spread tight%</div><div class="val" style="color:${tightPct > 70 ? '#44ff88' : '#ffaa00'}">${tightPct}%</div>
    <div class="lbl">Avg spread</div><div class="val">${spreads.length ? mean(spreads).toFixed(1) : '—'}</div>
    <div class="lbl">Min spread</div><div class="val">${spreads.length ? Math.min(...spreads).toFixed(0) : '—'}</div>
    <div class="lbl">Max spread</div><div class="val">${spreads.length ? Math.max(...spreads).toFixed(0) : '—'}</div>
  </div>`;

      h += `<div class="section-title">Fair Value Stability</div>`;
      h += `<div class="stat-grid">
    <div class="lbl">Ticks FV changed</div><div class="val">${fairChangePct}%</div>
    <div class="lbl">Avg FV move</div><div class="val">${fairChanges.length ? mean(fairChanges).toFixed(2) : '—'}</div>
    <div class="lbl">Max FV move</div><div class="val">${fairChanges.length ? Math.max(...fairChanges).toFixed(1) : '—'}</div>
  </div>`;
      h += `<div style="font-size:9px;color:#888;margin:2px 0">
    ${fairChangePct > 40 ? '⚠ Fair value moves frequently — mean-reversion strategy may be profitable.'
          : fairChangePct < 5 ? '✓ Fair value very stable — ideal for pure market-making.'
            : 'Fair value moderately stable — tight spreads are viable.'}
  </div>`;

      h += `<div class="section-title">Market Taker Behavior</div>`;
      h += `<div class="stat-grid">
    <div class="lbl">Avg mkt qty</div><div class="val">${mktTrades.length ? mean(mktTrades.map(t => t.qty)).toFixed(1) : '—'}</div>
    <div class="lbl">Big trades (≥5)</div><div class="val">${bigTrades.length}</div>
    <div class="lbl">Price impact (3T)</div><div class="val">${impacts.length ? mean(impacts).toFixed(3) : '—'}</div>
    <div class="lbl">Avg dist from FV</div><div class="val">${distFromFair.length ? mean(distFromFair.map(Math.abs)).toFixed(2) : '—'}</div>
  </div>`;
      h += `<div style="font-size:9px;color:#888;margin:2px 0">Price impact: avg FV change 3 ticks after a big taker trade (sign-adjusted). Positive = adverse selection risk.</div>`;

      if (snapsWithBotOrders.length) {
        h += `<div class="section-title">Your Bot Quote Levels (offset from FV)</div>`;
        h += `<div class="stat-grid">
      <div class="lbl">Avg bid offset</div><div class="val">${botBidLevels.length ? mean(botBidLevels).toFixed(2) : '—'}</div>
      <div class="lbl">Avg ask offset</div><div class="val">${botAskLevels.length ? mean(botAskLevels).toFixed(2) : '—'}</div>
      <div class="lbl">Ticks quoting</div><div class="val">${snapsWithBotOrders.length}</div>
      <div class="lbl">Fill rate</div><div class="val">${snapsWithBotOrders.length > 0 ? ((ownTrades.length / snapsWithBotOrders.length) * 100).toFixed(1) + '%' : '—'}</div>
    </div>`;
        h += `<div style="font-size:9px;color:#888;margin:2px 0">Fill rate = own trades / ticks with quotes. Low fill rate → quotes too far from market.</div>`;
      }

      if (snapsWithClearOrders.length) {
        h += `<div class="section-title" style="color:#4cc9f0">Your Bot Clear (Taking) Levels</div>`;
        h += `<div class="stat-grid">
      <div class="lbl">Avg clear bid offset</div><div class="val">${clearBidLevels.length ? mean(clearBidLevels).toFixed(2) : '—'}</div>
      <div class="lbl">Avg clear ask offset</div><div class="val">${clearAskLevels.length ? mean(clearAskLevels).toFixed(2) : '—'}</div>
      <div class="lbl">Ticks clearing</div><div class="val">${snapsWithClearOrders.length}</div>
    </div>`;
        h += `<div style="font-size:9px;color:#888;margin:2px 0">Clear actions = your bot taking/lifting existing resting orders. Negative bid offset = clearing below FV (buying cheap).</div>`;
      }

      h += `<div class="section-title">Qty Signatures (bot fingerprints)</div>`;
      h += `<div style="font-size:9px;margin:2px 0">`;
      for (const [qty, cnt] of topQtys) h += `<span style="color:#4cc9f0">${qty}</span>×${cnt} `;
      h += `</div>`;
      h += `<div style="font-size:9px;color:#888">Recurring qtys often reveal other bots' strategies (e.g. always trading in lots of 5 or 10).</div>`;

      h += `<div class="section-title">Daily Ranges</div>`;
      for (const [day, st] of Object.entries(dailyStats)) {
        const range = (st.max - st.min).toFixed(0);
        h += `<div style="font-size:9px;margin:1px 0">Day <b>${day}</b>: <span style="color:#4488ff">L=${st.min.toFixed(0)}</span>  <span style="color:#ff4444">H=${st.max.toFixed(0)}</span>  Range=<b>${range}</b>  Ticks=${st.count}</div>`;
      }

      h += `<div class="section-title">Trade Volume by Price-Level</div>`;
      h += `<div class="mini-chart" style="height:120px"><canvas id="vbpC"></canvas></div>`;
      h += `<div style="font-size:8px;color:#555">Volume profile: how much volume traded at each price. Blue=bought below FV, Red=sold above FV.</div>`;
      el.innerHTML = h;

      // Volume-by-price chart
      setTimeout(() => {
        const c = document.getElementById('vbpC'); if (!c) return;
        const { ctx, w, h } = setupCanvas(c, c.parentElement);
        if (!mktTrades.length) return;
        const prices = mktTrades.map(t => t.price);
        const mn = Math.min(...prices), mx = Math.max(...prices), rng = mx - mn || 1;
        const bins = 30; const bw = (mx - mn) / bins || 1;
        const buyVol = new Array(bins).fill(0), sellVol = new Array(bins).fill(0);
        for (const t of mktTrades) {
          const s = findNearest(snaps, t.ts);
          const fv = s ? s.fair : null;
          let b = Math.floor((t.price - mn) / bw); if (b >= bins) b = bins - 1; if (b < 0) b = 0;
          if (fv != null && t.price >= fv) sellVol[b] += t.qty;
          else buyVol[b] += t.qty;
        }
        const maxV = Math.max(...buyVol, ...sellVol, 1);
        const barH = (h - 4) / bins;
        for (let i = 0; i < bins; i++) {
          const y = i * barH; const price = mn + i * bw;
          if (buyVol[i]) { ctx.fillStyle = 'rgba(68,136,255,0.65)'; ctx.fillRect(0, y, (buyVol[i] / maxV) * (w / 2), barH - 1); }
          if (sellVol[i]) { ctx.fillStyle = 'rgba(255,68,68,0.65)'; ctx.fillRect(w / 2, y, (sellVol[i] / maxV) * (w / 2), barH - 1); }
        }
        ctx.fillStyle = '#444'; ctx.font = '7px Courier New';
        ctx.fillText(mn.toFixed(0), 2, h - 2); ctx.textAlign = 'right'; ctx.fillText(mx.toFixed(0), w - 2, h - 2); ctx.textAlign = 'start';
        ctx.strokeStyle = '#555'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke();
      }, 30);
    }
