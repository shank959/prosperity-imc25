// =====================================================================
    // SECTION 21b — STRATEGY ANALYSIS HELPERS
    // =====================================================================
    function computeStrategyAnalysis(snaps, trades) {
      const posLimit = PRODUCT_CONFIG[currentProduct]?.posLimit || 80;
      const own = trades.filter(t => t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION');

      // Fill edges
      const buyEdges = [], sellEdges = [];
      for (const t of own) {
        const s = findNearest(snaps, t.ts);
        if (!s || s.fair == null) continue;
        const edge = +(t.price - s.fair).toFixed(2);
        if (t.buyer === 'SUBMISSION') buyEdges.push({ ts: t.ts, edge });
        else sellEdges.push({ ts: t.ts, edge });
      }
      const buysBelowFV = buyEdges.filter(e => e.edge < 0).length;
      const sellsAboveFV = sellEdges.filter(e => e.edge > 0).length;
      const buysBelowFVPct = buyEdges.length ? +(buysBelowFV / buyEdges.length * 100).toFixed(1) : 0;
      const sellsAboveFVPct = sellEdges.length ? +(sellsAboveFV / sellEdges.length * 100).toFixed(1) : 0;
      const avgBuyEdge = buyEdges.length ? +mean(buyEdges.map(e => e.edge)).toFixed(3) : 0;
      const avgSellEdge = sellEdges.length ? +mean(sellEdges.map(e => e.edge)).toFixed(3) : 0;
      // Adverse selection: 0=perfect, 100=worst. Buy above FV or sell below FV is adverse.
      const adverseBuys = buyEdges.length ? buyEdges.filter(e => e.edge > 0).length / buyEdges.length : 0;
      const adverseSells = sellEdges.length ? sellEdges.filter(e => e.edge < 0).length / sellEdges.length : 0;
      const adverseSelectionScore = +((adverseBuys + adverseSells) / 2 * 100).toFixed(1);

      // Position patterns
      const positions = snaps.map(s => s.pos).filter(p => p != null);
      let zeroCrossings = 0;
      for (let i = 1; i < positions.length; i++) {
        if ((positions[i - 1] > 0 && positions[i] <= 0) || (positions[i - 1] < 0 && positions[i] >= 0)) zeroCrossings++;
      }
      const nDays = Math.max(1, Store.days.length);
      const zeroCrossingsPerDay = +(zeroCrossings / nDays).toFixed(1);
      // Sawtooth score: zero crossings per 1000 ticks, scaled 0-100
      const sawtoothRaw = positions.length > 100 ? zeroCrossings / (positions.length / 1000) : 0;
      const sawtoothScore = +Math.min(100, sawtoothRaw * 10).toFixed(1);
      // Rolling mean drift
      const windowSize = Math.min(200, Math.floor(positions.length / 5));
      let driftSum = 0;
      const rollingMean = [];
      for (let i = 0; i < positions.length; i++) {
        const start = Math.max(0, i - windowSize + 1);
        const slice = positions.slice(start, i + 1);
        const rm = mean(slice);
        rollingMean.push(rm);
        driftSum += Math.abs(rm);
      }
      const avgAbsDrift = positions.length ? +(driftSum / positions.length).toFixed(1) : 0;
      const overallDrift = positions.length ? mean(positions) : 0;
      const driftDirection = overallDrift > posLimit * 0.1 ? 'long-biased' : overallDrift < -posLimit * 0.1 ? 'short-biased' : 'balanced';
      const longLimitPct = positions.length ? +(positions.filter(p => p >= posLimit * 0.9).length / positions.length * 100).toFixed(1) : 0;
      const shortLimitPct = positions.length ? +(positions.filter(p => p <= -posLimit * 0.9).length / positions.length * 100).toFixed(1) : 0;

      // PnL slope analysis
      const pnls = snaps.map(s => s.pnl).filter(p => p != null);
      let overallSlope = 0;
      if (pnls.length > 1) {
        // Simple linear regression slope
        const n = pnls.length;
        let sx = 0, sy = 0, sxy = 0, sx2 = 0;
        for (let i = 0; i < n; i++) { sx += i; sy += pnls[i]; sxy += i * pnls[i]; sx2 += i * i; }
        overallSlope = +((n * sxy - sx * sy) / (n * sx2 - sx * sx)).toFixed(4);
      }
      // Segment analysis: divide into 1000-tick windows
      const segSize = 1000;
      let steadyPeriods = 0, flatPeriods = 0, adversePeriods = 0;
      const pnlSnaps = snaps.filter(s => s.pnl != null);
      for (let i = 0; i + segSize <= pnlSnaps.length; i += segSize) {
        const segStart = pnlSnaps[i].pnl, segEnd = pnlSnaps[i + segSize - 1].pnl;
        const slope = (segEnd - segStart) / segSize;
        if (slope > 0.01) steadyPeriods++;
        else if (slope < -0.01) adversePeriods++;
        else flatPeriods++;
      }
      // Max drawdown duration
      let maxPnl = 0, maxDD = 0, ddStartIdx = 0, maxDDDuration = 0, ddTroughIdx = 0;
      let peakIdx = 0;
      for (let i = 0; i < pnlSnaps.length; i++) {
        if (pnlSnaps[i].pnl > maxPnl) { maxPnl = pnlSnaps[i].pnl; peakIdx = i; }
        const dd = maxPnl - pnlSnaps[i].pnl;
        if (dd > maxDD) { maxDD = dd; ddTroughIdx = i; maxDDDuration = pnlSnaps[i].ts - pnlSnaps[peakIdx].ts; }
      }
      // Recovery time: from trough back to peak level
      let recoveryTime = null;
      for (let i = ddTroughIdx + 1; i < pnlSnaps.length; i++) {
        if (pnlSnaps[i].pnl >= maxPnl) { recoveryTime = pnlSnaps[i].ts - pnlSnaps[ddTroughIdx].ts; break; }
      }

      // Spread danger zones
      const spreadData = snaps.filter(s => s.bids.length && s.asks.length)
        .map(s => ({ ts: s.ts, spread: Math.min(...s.asks.map(a => a.price)) - Math.max(...s.bids.map(b => b.price)) }));
      const spreads = spreadData.map(d => d.spread);
      const sp25 = spreads.length ? pctile(spreads, 25) : 0;
      const sp75 = spreads.length ? pctile(spreads, 75) : Infinity;
      const spMean = spreads.length ? mean(spreads) : 0;
      const spMedian = spreads.length ? med(spreads) : 0;
      // Find contiguous tight/wide zones
      function findZones(data, test) {
        const zones = []; let start = null, vals = [];
        for (const d of data) {
          if (test(d.spread)) {
            if (start == null) start = d.ts;
            vals.push(d.spread);
          } else if (start != null) {
            zones.push({ startTs: start, endTs: d.ts, avgSpread: +mean(vals).toFixed(2) });
            start = null; vals = [];
          }
        }
        if (start != null) zones.push({ startTs: start, endTs: data[data.length - 1].ts, avgSpread: +mean(vals).toFixed(2) });
        return zones;
      }
      const tightZones = findZones(spreadData, sp => sp < sp25);
      const wideZones = findZones(spreadData, sp => sp > sp75);
      // % fills during tight spreads
      let fillsInTight = 0;
      for (const t of own) {
        for (const z of tightZones) {
          if (t.ts >= z.startTs && t.ts <= z.endTs) { fillsInTight++; break; }
        }
      }
      const tightSpreadFillRate = own.length ? +(fillsInTight / own.length * 100).toFixed(1) : 0;

      // Trade qty signatures
      const mkt = trades.filter(t => t.buyer !== 'SUBMISSION' && t.seller !== 'SUBMISSION');
      const qtyFreq = {};
      for (const t of mkt) qtyFreq[t.qty] = (qtyFreq[t.qty] || 0) + 1;
      const distribution = Object.entries(qtyFreq).sort((a, b) => b[1] - a[1])
        .map(([q, c]) => ({ qty: +q, count: c, pctOfTotal: +(c / mkt.length * 100).toFixed(1) }));
      const botCluster = distribution.filter(d => d.qty >= 2 && d.qty <= 5);
      const informedSizes = distribution.filter(d => d.qty >= 6);
      const singleTradeCount = qtyFreq[1] || 0;

      return {
        posLimit,
        fillAnalysis: { buyEdges, sellEdges, buysBelowFVPct, sellsAboveFVPct, avgBuyEdge, avgSellEdge, adverseSelectionScore },
        positionPatterns: { positions, rollingMean, zeroCrossings, zeroCrossingsPerDay, sawtoothScore, avgAbsDrift, driftDirection, timeAtLimit: { long: longLimitPct, short: shortLimitPct } },
        pnlSlope: { pnls: pnlSnaps, overallPerTick: overallSlope, steadyPeriods, flatPeriods, adversePeriods, maxDrawdownDuration: maxDDDuration, recoveryTime, maxDD },
        spreadAnalysis: { spreadData, spreads, sp25, sp75, spMean, spMedian, tightZones, wideZones, tightSpreadFillRate },
        tradeQtySignatures: { distribution, botCluster, informedSizes, singleTradeCount }
      };
    }

    // =====================================================================
    // SECTION 21c — STRATEGY TAB RENDERING
    // =====================================================================
    function renderStrategyTab() {
      const el = document.getElementById('strategyTab');
      const snaps = getVisSnaps();
      const trades = Store.trades.filter(t => t.symbol === currentProduct);
      if (snaps.length < 50) { el.innerHTML = '<div style="color:#555;padding:8px">Need more data (50+ snapshots)</div>'; return; }

      const analysis = computeStrategyAnalysis(snaps, trades);
      const fa = analysis.fillAnalysis, pp = analysis.positionPatterns, ps = analysis.pnlSlope, sa = analysis.spreadAnalysis, tq = analysis.tradeQtySignatures;

      let h = '<div class="strategy-charts">';

      h += `<div class="explain-box"><b>Strategy health overview</b> for the visible time window. <b>PnL</b>: green=rising, red=falling, shaded=drawdown. <b>Position</b>: high sawtooth score = healthy mean-reversion (crosses zero often). <b>Fills</b>: <span class="good">buys below FV &amp; sells above FV = good edge</span>. <b>Spread</b>: fills during tight spreads = <span class="bad">adverse selection risk</span>. <b>Qty dist</b>: cyan=bot cluster (2–5), yellow=large/informed trades.</div>`;

      // PnL chart section
      h += '<div class="chart-section">';
      h += '<div class="section-title">PnL Over Time — slope: green rising, red falling, shaded = drawdown</div>';
      h += `<div class="chart-label"><span>Slope/tick: <span class="val">${ps.overallPerTick}</span></span><span>Drawdown: <span class="${ps.maxDD > 500 ? 'bad' : 'good'}">${ps.maxDD.toFixed(0)}</span></span></div>`;
      h += '<div class="mini-chart" style="height:120px"><canvas id="stratPnlC"></canvas></div>';
      h += `<div class="chart-label"><span>Up: <span class="good">${ps.steadyPeriods}</span> Flat: <span class="val">${ps.flatPeriods}</span> Down: <span class="bad">${ps.adversePeriods}</span></span></div>`;
      h += '</div>';

      // Position chart section
      h += '<div class="chart-section">';
      h += '<div class="section-title">Position Trace — yellow=pos, dashed=rolling mean, dots=zero crossings</div>';
      h += `<div class="chart-label"><span>Sawtooth: <span class="${pp.sawtoothScore > 30 ? 'good' : 'bad'}">${pp.sawtoothScore}</span>/100</span><span>Drift: <span class="val">${pp.driftDirection}</span></span></div>`;
      h += '<div class="mini-chart" style="height:100px"><canvas id="stratPosC"></canvas></div>';
      h += `<div class="chart-label"><span>Zero-X: <span class="val">${pp.zeroCrossings}</span> (${pp.zeroCrossingsPerDay}/day)</span><span>@Limit: L<span class="${pp.timeAtLimit.long > 10 ? 'bad' : 'val'}">${pp.timeAtLimit.long}%</span> S<span class="${pp.timeAtLimit.short > 10 ? 'bad' : 'val'}">${pp.timeAtLimit.short}%</span></span></div>`;
      h += '</div>';

      // Fill scatter section
      h += '<div class="chart-section">';
      h += '<div class="section-title">Fill Scatter vs FV — blue=buy, orange=sell, zero line=fair value</div>';
      h += `<div class="chart-label"><span>Buys&lt;FV: <span class="${fa.buysBelowFVPct > 50 ? 'good' : 'bad'}">${fa.buysBelowFVPct}%</span></span><span>Sells&gt;FV: <span class="${fa.sellsAboveFVPct > 50 ? 'good' : 'bad'}">${fa.sellsAboveFVPct}%</span></span><span>Adv.Sel: <span class="${fa.adverseSelectionScore < 40 ? 'good' : 'bad'}">${fa.adverseSelectionScore}</span></span></div>`;
      h += '<div class="mini-chart" style="height:100px"><canvas id="stratFillC"></canvas></div>';
      h += `<div class="chart-label"><span>AvgBuyEdge: <span class="${fa.avgBuyEdge < 0 ? 'good' : 'bad'}">${fa.avgBuyEdge}</span></span><span>AvgSellEdge: <span class="${fa.avgSellEdge > 0 ? 'good' : 'bad'}">${fa.avgSellEdge}</span></span></div>`;
      h += '</div>';

      // Spread timeline section
      h += '<div class="chart-section">';
      h += '<div class="section-title">Spread Over Time — red zones=tight (danger), green zones=wide (opportunity)</div>';
      h += `<div class="chart-label"><span>Mean: <span class="val">${sa.spMean.toFixed(2)}</span></span><span>Median: <span class="val">${sa.spMedian.toFixed(2)}</span></span><span>Fills@tight: <span class="${sa.tightSpreadFillRate > 30 ? 'bad' : 'good'}">${sa.tightSpreadFillRate}%</span></span></div>`;
      h += '<div class="mini-chart" style="height:80px"><canvas id="stratSpreadC"></canvas></div>';
      h += '</div>';

      // Trade quantity distribution
      h += '<div class="chart-section">';
      h += '<div class="section-title">Trade Qty Distribution — grey=qty1, cyan=bot(2–5), yellow=large(6+)</div>';
      h += `<div class="chart-label"><span>Bot cluster (2-5): <span class="val">${tq.botCluster.reduce((s, d) => s + d.count, 0)}</span></span><span>Qty 1: <span class="val">${tq.singleTradeCount}</span></span><span>Qty 6+: <span class="${tq.informedSizes.length ? 'bad' : 'val'}">${tq.informedSizes.reduce((s, d) => s + d.count, 0)}</span></span></div>`;
      h += '<div class="mini-chart" style="height:80px"><canvas id="stratQtyC"></canvas></div>';
      h += '</div>';

      h += '</div>';
      el.innerHTML = h;

      // Render canvases after DOM updates
      setTimeout(() => {
        renderStratPnl(analysis);
        renderStratPosition(analysis);
        renderStratFills(analysis);
        renderStratSpread(analysis);
        renderStratQty(analysis);
      }, 30);
    }

    function renderStratPnl(analysis) {
      const c = document.getElementById('stratPnlC'); if (!c) return;
      const { ctx, w, h } = setupCanvas(c, c.parentElement);
      const pnlSnaps = analysis.pnlSlope.pnls;
      if (!pnlSnaps.length) return;

      const pnls = pnlSnaps.map(s => s.pnl);
      const pMin = Math.min(...pnls) - 50, pMax = Math.max(...pnls) + 50;
      const tMin = pnlSnaps[0].ts, tMax = pnlSnaps[pnlSnaps.length - 1].ts;
      const tRange = tMax - tMin || 1, pRange = pMax - pMin || 1;
      const tx = ts => ((ts - tMin) / tRange) * w;
      const py = p => h - ((p - pMin) / pRange) * h;

      // Drawdown shading
      let peak = pnls[0];
      ctx.fillStyle = 'rgba(255,50,50,0.15)';
      for (let i = 1; i < pnlSnaps.length; i++) {
        if (pnlSnaps[i].pnl > peak) peak = pnlSnaps[i].pnl;
        if (pnlSnaps[i].pnl < peak) {
          const x = tx(pnlSnaps[i].ts);
          ctx.fillRect(x - 0.5, py(peak), 1, py(pnlSnaps[i].pnl) - py(peak));
        }
      }

      // Peak watermark line
      peak = pnls[0];
      ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 0.5; ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (let i = 0; i < pnlSnaps.length; i++) {
        if (pnlSnaps[i].pnl > peak) peak = pnlSnaps[i].pnl;
        const x = tx(pnlSnaps[i].ts), y = py(peak);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.setLineDash([]);

      // PnL line with slope-based coloring
      ctx.lineWidth = 1.5;
      for (let i = 1; i < pnlSnaps.length; i++) {
        const slope = pnlSnaps[i].pnl - pnlSnaps[i - 1].pnl;
        ctx.strokeStyle = slope > 0.5 ? '#44ff44' : slope < -0.5 ? '#ff4444' : '#888';
        ctx.beginPath();
        ctx.moveTo(tx(pnlSnaps[i - 1].ts), py(pnlSnaps[i - 1].pnl));
        ctx.lineTo(tx(pnlSnaps[i].ts), py(pnlSnaps[i].pnl));
        ctx.stroke();
      }

      // Labels
      ctx.fillStyle = '#444'; ctx.font = '7px Courier New';
      ctx.fillText(pMin.toFixed(0), 2, h - 2);
      ctx.textAlign = 'right'; ctx.fillText(pMax.toFixed(0), w - 2, 9); ctx.textAlign = 'start';
    }

    function renderStratPosition(analysis) {
      const c = document.getElementById('stratPosC'); if (!c) return;
      const { ctx, w, h } = setupCanvas(c, c.parentElement);
      const positions = analysis.positionPatterns.positions;
      const rollingMean = analysis.positionPatterns.rollingMean;
      if (!positions.length) return;

      const posLimit = analysis.posLimit;
      const yMin = -posLimit - 5, yMax = posLimit + 5, yRange = yMax - yMin;
      const posToY = p => h - ((p - yMin) / yRange) * h;
      const snaps = getVisSnaps().filter(s => s.pos != null);
      const tMin = snaps[0]?.ts || 0, tMax = snaps[snaps.length - 1]?.ts || 1, tRange = tMax - tMin || 1;
      const tx = ts => ((ts - tMin) / tRange) * w;

      // Limit bands
      ctx.fillStyle = 'rgba(255,60,60,0.08)';
      ctx.fillRect(0, 0, w, posToY(posLimit * 0.75));
      ctx.fillRect(0, posToY(-posLimit * 0.75), w, h - posToY(-posLimit * 0.75));

      // Zero line
      ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5;
      const y0 = posToY(0);
      ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(w, y0); ctx.stroke();

      // Limit dashed lines
      ctx.strokeStyle = 'rgba(255,60,60,0.3)'; ctx.setLineDash([2, 3]);
      [posLimit, -posLimit].forEach(lim => { ctx.beginPath(); ctx.moveTo(0, posToY(lim)); ctx.lineTo(w, posToY(lim)); ctx.stroke(); });
      ctx.setLineDash([]);

      // Position line
      ctx.strokeStyle = '#ffcc00'; ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = 0; i < snaps.length; i++) {
        const x = tx(snaps[i].ts), y = posToY(snaps[i].pos);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Rolling mean drift line
      if (rollingMean.length === snaps.length) {
        ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 0.8; ctx.setLineDash([4, 3]);
        ctx.beginPath();
        for (let i = 0; i < snaps.length; i++) {
          const x = tx(snaps[i].ts), y = posToY(rollingMean[i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.setLineDash([]);
      }

      // Zero-crossing markers
      for (let i = 1; i < snaps.length; i++) {
        if ((snaps[i - 1].pos > 0 && snaps[i].pos <= 0) || (snaps[i - 1].pos < 0 && snaps[i].pos >= 0)) {
          drawDot(ctx, tx(snaps[i].ts), y0, 2.5, '#4cc9f0', 0.7);
        }
      }

      // Labels
      ctx.fillStyle = '#444'; ctx.font = '7px Courier New';
      ctx.fillText('+' + posLimit, 2, posToY(posLimit) + 8);
      ctx.fillText('-' + posLimit, 2, posToY(-posLimit) - 2);
    }

    function renderStratFills(analysis) {
      const c = document.getElementById('stratFillC'); if (!c) return;
      const { ctx, w, h } = setupCanvas(c, c.parentElement);
      const buyEdges = analysis.fillAnalysis.buyEdges;
      const sellEdges = analysis.fillAnalysis.sellEdges;
      const allEdges = [...buyEdges, ...sellEdges];
      if (!allEdges.length) { ctx.fillStyle = '#555'; ctx.font = '9px Courier New'; ctx.fillText('No fills', 10, h / 2); return; }

      const edgeVals = allEdges.map(e => e.edge);
      const eMax = Math.max(Math.abs(Math.min(...edgeVals)), Math.abs(Math.max(...edgeVals)), 1);
      const eRange = eMax * 2;
      const tMin = Math.min(...allEdges.map(e => e.ts)), tMax = Math.max(...allEdges.map(e => e.ts));
      const tRange = tMax - tMin || 1;
      const tx = ts => ((ts - tMin) / tRange) * w;
      const ey = edge => h / 2 - (edge / eMax) * (h / 2 - 6);

      // Zero line (FV reference)
      ctx.strokeStyle = COLORS.fairLine; ctx.lineWidth = 0.8; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
      ctx.setLineDash([]);

      // Shade: above zero = adverse for buys, below zero = adverse for sells
      ctx.fillStyle = 'rgba(255,50,50,0.05)';
      ctx.fillRect(0, 0, w, h / 2);
      ctx.fillStyle = 'rgba(50,255,50,0.05)';
      ctx.fillRect(0, h / 2, w, h / 2);

      // Buy fills (blue dots)
      for (const e of buyEdges) {
        drawDot(ctx, tx(e.ts), ey(e.edge), 3, '#4488ff', 0.7);
      }
      // Sell fills (orange dots)
      for (const e of sellEdges) {
        drawDot(ctx, tx(e.ts), ey(e.edge), 3, '#ff8800', 0.7);
      }

      // Rolling average edge (window of 20 fills)
      if (allEdges.length > 5) {
        const sorted = [...allEdges].sort((a, b) => a.ts - b.ts);
        const winSize = Math.min(20, Math.floor(sorted.length / 3));
        ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1; ctx.setLineDash([3, 2]);
        ctx.beginPath();
        let started = false;
        for (let i = winSize; i < sorted.length; i++) {
          const slice = sorted.slice(i - winSize, i);
          const avg = mean(slice.map(e => e.edge));
          const x = tx(sorted[i].ts), y = ey(avg);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.setLineDash([]);
      }

      // Labels
      ctx.fillStyle = '#444'; ctx.font = '7px Courier New';
      ctx.fillText('+' + eMax.toFixed(1), 2, 8);
      ctx.fillText('-' + eMax.toFixed(1), 2, h - 2);
      ctx.fillStyle = '#4488ff'; ctx.fillText('Buy', w - 50, 8);
      ctx.fillStyle = '#ff8800'; ctx.fillText('Sell', w - 25, 8);
    }

    function renderStratSpread(analysis) {
      const c = document.getElementById('stratSpreadC'); if (!c) return;
      const { ctx, w, h } = setupCanvas(c, c.parentElement);
      const sd = analysis.spreadAnalysis.spreadData;
      if (!sd.length) return;

      const spreads = sd.map(d => d.spread);
      const sMin = Math.max(0, Math.min(...spreads) - 0.5), sMax = Math.max(...spreads) + 0.5;
      const sRange = sMax - sMin || 1;
      const tMin = sd[0].ts, tMax = sd[sd.length - 1].ts, tRange = tMax - tMin || 1;
      const tx = ts => ((ts - tMin) / tRange) * w;
      const sy = s => h - ((s - sMin) / sRange) * h;
      const sp25 = analysis.spreadAnalysis.sp25, sp75 = analysis.spreadAnalysis.sp75;

      // Danger zones (tight spread)
      for (const z of analysis.spreadAnalysis.tightZones) {
        const x1 = tx(z.startTs), x2 = tx(z.endTs);
        ctx.fillStyle = 'rgba(255,50,50,0.12)';
        ctx.fillRect(x1, 0, Math.max(1, x2 - x1), h);
      }
      // Opportunity zones (wide spread)
      for (const z of analysis.spreadAnalysis.wideZones) {
        const x1 = tx(z.startTs), x2 = tx(z.endTs);
        ctx.fillStyle = 'rgba(50,255,50,0.08)';
        ctx.fillRect(x1, 0, Math.max(1, x2 - x1), h);
      }

      // Mean and median dashed lines
      ctx.strokeStyle = 'rgba(76,201,240,0.4)'; ctx.lineWidth = 0.5; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(0, sy(analysis.spreadAnalysis.spMean)); ctx.lineTo(w, sy(analysis.spreadAnalysis.spMean)); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.beginPath(); ctx.moveTo(0, sy(analysis.spreadAnalysis.spMedian)); ctx.lineTo(w, sy(analysis.spreadAnalysis.spMedian)); ctx.stroke();
      ctx.setLineDash([]);

      // Spread line
      ctx.lineWidth = 1;
      for (let i = 1; i < sd.length; i++) {
        const sp = sd[i].spread;
        ctx.strokeStyle = sp < sp25 ? '#ff4444' : sp > sp75 ? '#44ff44' : '#aaa';
        ctx.beginPath();
        ctx.moveTo(tx(sd[i - 1].ts), sy(sd[i - 1].spread));
        ctx.lineTo(tx(sd[i].ts), sy(sd[i].spread));
        ctx.stroke();
      }

      // Labels
      ctx.fillStyle = '#444'; ctx.font = '7px Courier New';
      ctx.fillText(sMin.toFixed(1), 2, h - 2);
      ctx.textAlign = 'right'; ctx.fillText(sMax.toFixed(1), w - 2, 9); ctx.textAlign = 'start';
    }

    function renderStratQty(analysis) {
      const c = document.getElementById('stratQtyC'); if (!c) return;
      const { ctx, w, h } = setupCanvas(c, c.parentElement);
      const dist = analysis.tradeQtySignatures.distribution;
      if (!dist.length) return;

      // Show top 15 quantities, sorted by qty value
      const shown = [...dist].sort((a, b) => a.qty - b.qty).slice(0, 15);
      const maxCount = Math.max(...shown.map(d => d.count));
      const barH = Math.min(14, (h - 4) / shown.length);
      const labelW = 22, barAreaW = w - labelW - 4;

      for (let i = 0; i < shown.length; i++) {
        const d = shown[i];
        const y = 2 + i * barH;
        const bw = (d.count / maxCount) * barAreaW;

        // Color by category
        if (d.qty === 1) ctx.fillStyle = '#666';
        else if (d.qty >= 2 && d.qty <= 5) ctx.fillStyle = '#4cc9f0';
        else ctx.fillStyle = '#ffdd44';

        ctx.fillRect(labelW, y, bw, barH - 2);

        // Qty label
        ctx.fillStyle = '#888'; ctx.font = '7px Courier New'; ctx.textAlign = 'right';
        ctx.fillText(d.qty, labelW - 2, y + barH - 3);

        // Count label on bar
        if (bw > 25) {
          ctx.fillStyle = '#000'; ctx.textAlign = 'left';
          ctx.fillText(d.count, labelW + 2, y + barH - 3);
        }
      }
      ctx.textAlign = 'start';

      // Legend
      ctx.font = '6px Courier New';
      ctx.fillStyle = '#666'; ctx.fillText('grey=1', w - 80, h - 2);
      ctx.fillStyle = '#4cc9f0'; ctx.fillText('cyan=2-5', w - 55, h - 2);
      ctx.fillStyle = '#ffdd44'; ctx.fillText('yel=6+', w - 25, h - 2);
    }
