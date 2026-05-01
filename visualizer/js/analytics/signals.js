// =====================================================================
    // SECTION 21d — SIGNALS TAB (spike detection + trader informativeness)
    // =====================================================================
    function computeSignalsAnalysis(snaps, trades) {
      // 1. Compute best bid/ask deltas between consecutive snapshots
      const deltas = [];
      for (let i = 1; i < snaps.length; i++) {
        if (!snaps[i].bids.length || !snaps[i-1].bids.length) continue;
        if (!snaps[i].asks.length || !snaps[i-1].asks.length) continue;
        const prevBB = Math.max(...snaps[i-1].bids.map(b => b.price));
        const currBB = Math.max(...snaps[i].bids.map(b => b.price));
        const prevBA = Math.min(...snaps[i-1].asks.map(a => a.price));
        const currBA = Math.min(...snaps[i].asks.map(a => a.price));
        deltas.push({
          ts: snaps[i].ts,
          bidDelta: currBB - prevBB,
          askDelta: currBA - prevBA,
          bestBid: currBB,
          bestAsk: currBA,
          wallMid: snaps[i].fair
        });
      }

      // 2. Compute stats for threshold calibration
      const bidDeltas = deltas.map(d => d.bidDelta);
      const askDeltas = deltas.map(d => d.askDelta);
      const bidStd = std(bidDeltas.map(Math.abs));
      const askStd = std(askDeltas.map(Math.abs));
      const combinedStd = Math.max(bidStd, askStd);

      // 3. Spike detection function
      function detectSpikes(threshold) {
        const spikes = [];
        for (const d of deltas) {
          const bidBig = Math.abs(d.bidDelta) > threshold;
          const askBig = Math.abs(d.askDelta) > threshold;
          if (!bidBig && !askBig) continue;
          // Pick the larger delta
          const primary = Math.abs(d.bidDelta) >= Math.abs(d.askDelta) ? 'bid' : 'ask';
          const delta = primary === 'bid' ? d.bidDelta : d.askDelta;
          spikes.push({
            ts: d.ts,
            side: primary,
            direction: delta > 0 ? 'up' : 'down',
            magnitude: Math.abs(delta),
            bidDelta: d.bidDelta,
            askDelta: d.askDelta,
            wallMid: d.wallMid,
            bestBid: d.bestBid,
            bestAsk: d.bestAsk
          });
        }
        return spikes;
      }

      // 4. Measure spike outcomes at various lookaheads
      function measureOutcomes(spikes, lookaheads) {
        return spikes.map(sp => {
          const idx = snaps.findIndex(s => s.ts === sp.ts);
          if (idx < 0) return { ...sp, outcomes: {} };
          const outcomes = {};
          for (const n of lookaheads) {
            const fi = idx + n;
            if (fi < snaps.length && snaps[fi].fair != null && snaps[idx].fair != null) {
              outcomes[n] = snaps[fi].fair - snaps[idx].fair;
            }
          }
          return { ...sp, outcomes };
        });
      }

      // 5. Trader analysis
      const lookaheads = [1, 5, 10, 50, 100];
      const mktTrades = trades.filter(t => t.buyer !== 'SUBMISSION' && t.seller !== 'SUBMISSION');
      const traderNames = new Set();
      for (const t of mktTrades) {
        if (t.buyer && t.buyer.trim()) traderNames.add(t.buyer.trim());
        if (t.seller && t.seller.trim()) traderNames.add(t.seller.trim());
      }
      const hasTraderNames = traderNames.size > 0;

      const traderStats = {};
      if (hasTraderNames) {
        for (const name of traderNames) {
          const tTrades = mktTrades.filter(t => t.buyer === name || t.seller === name);
          if (tTrades.length < 2) continue;
          const stats = { count: tTrades.length, totalQty: tTrades.reduce((s, t) => s + (t.qty || 0), 0), hitRates: {} };
          for (const n of lookaheads) {
            let correct = 0, total = 0;
            for (const t of tTrades) {
              const snap = findNearest(snaps, t.ts);
              const futureSnap = findNearest(snaps, t.ts + n * 100);
              if (!snap || !futureSnap || snap.fair == null || futureSnap.fair == null) continue;
              if (futureSnap.ts <= snap.ts) continue;
              const move = futureSnap.fair - snap.fair;
              if (Math.abs(move) < 0.01) continue;
              const isBuy = t.buyer === name;
              const predicted = isBuy ? move > 0 : move < 0;
              total++;
              if (predicted) correct++;
            }
            stats.hitRates[n] = { correct, total, rate: total > 0 ? +(correct / total * 100).toFixed(1) : null };
          }
          const hr10 = stats.hitRates[10];
          stats.classification = (hr10 && hr10.rate !== null && hr10.rate > 55 && hr10.total >= 5) ? 'informed' : 'noise';
          stats.avgQty = stats.count > 0 ? +(stats.totalQty / stats.count).toFixed(1) : 0;
          traderStats[name] = stats;
        }
      }

      return { deltas, bidStd, askStd, combinedStd, detectSpikes, measureOutcomes, traderStats, hasTraderNames, lookaheads, snaps };
    }

    function renderSignalsTab() {
      const el = document.getElementById('signalsTab');
      const snaps = getVisSnaps();
      const trades = Store.trades.filter(t => t.symbol === currentProduct);
      if (snaps.length < 50) { el.innerHTML = '<div style="color:#555;padding:8px">Need more data (50+ snapshots)</div>'; return; }

      if (!signalsCache) {
        el.innerHTML = '<div style="color:#4cc9f0;padding:8px">Computing signal analysis...</div>';
        setTimeout(() => {
          signalsCache = computeSignalsAnalysis(snaps, trades);
          renderSignalsTabInner(el);
        }, 30);
      } else {
        renderSignalsTabInner(el);
      }
    }

    function renderSignalsTabInner(el) {
      const sc = signalsCache;
      const defaultThreshold = +(sc.combinedStd * 2).toFixed(2);
      const spikes = sc.detectSpikes(defaultThreshold);
      const spikesWithOutcomes = sc.measureOutcomes(spikes, sc.lookaheads);

      let h = '<div class="strategy-charts">';
      h += `<div class="explain-box"><b>Spike Detection:</b> finds timestamps where best bid/ask moves sharply. <b>Trader Leaderboard:</b> for each counterparty, shows directional prediction accuracy (hit rate) — does their trade direction predict future wall_mid moves? <span class="warn">Trader names require output.log or local sandbox data.</span></div>`;

      // Spike controls
      h += '<div class="chart-section">';
      h += '<div class="section-title">Spike Detection Controls</div>';
      h += `<div class="stat-grid">
        <div class="lbl">Threshold</div><div class="val"><input type="range" id="spikeThreshold" min="0.5" max="20" step="0.5" value="${defaultThreshold}" style="width:80px"><span id="spikeThreshLabel">${defaultThreshold}</span></div>
        <div class="lbl">Bid delta std</div><div class="val">${sc.bidStd.toFixed(3)}</div>
        <div class="lbl">Ask delta std</div><div class="val">${sc.askStd.toFixed(3)}</div>
        <div class="lbl">Default (2*std)</div><div class="val">${defaultThreshold}</div>
        <div class="lbl">Events found</div><div class="val" id="spikeCount">${spikes.length}</div>
      </div>`;
      h += '</div>';

      // Spike predictiveness summary
      h += '<div class="chart-section">';
      h += '<div class="section-title">Spike Predictiveness</div>';
      h += '<div style="font-size:9px;overflow-x:auto">';
      h += '<table style="width:100%;border-collapse:collapse;font-size:9px">';
      h += '<tr style="color:#888"><th>Side</th><th>Dir</th><th>Count</th>';
      for (const n of sc.lookaheads) h += `<th>Hit@${n}</th>`;
      h += '</tr>';
      for (const side of ['bid', 'ask']) {
        for (const dir of ['up', 'down']) {
          const subset = spikesWithOutcomes.filter(s => s.side === side && s.direction === dir);
          if (!subset.length) continue;
          h += `<tr><td style="color:${side === 'bid' ? '#4488ff' : '#ff4444'}">${side}</td><td>${dir === 'up' ? '&#9650;' : '&#9660;'}</td><td>${subset.length}</td>`;
          for (const n of sc.lookaheads) {
            const moves = subset.filter(s => s.outcomes[n] != null).map(s => s.outcomes[n]);
            if (!moves.length) { h += '<td>—</td>'; continue; }
            const hitRate = dir === 'up'
              ? (moves.filter(m => m > 0).length / moves.length * 100)
              : (moves.filter(m => m < 0).length / moves.length * 100);
            const cls = hitRate > 55 ? 'good' : hitRate < 45 ? 'bad' : 'val';
            h += `<td class="${cls}">${hitRate.toFixed(0)}%</td>`;
          }
          h += '</tr>';
        }
      }
      h += '</table></div>';
      h += '</div>';

      // Spike timeline chart
      h += '<div class="chart-section">';
      h += '<div class="section-title">Spike Timeline — green=up, red=down, size=magnitude</div>';
      h += '<div class="mini-chart" style="height:120px"><canvas id="sigTimelineC"></canvas></div>';
      h += '</div>';

      // Spike scatter chart
      h += '<div class="chart-section">';
      h += '<div class="section-title">Spike Magnitude vs Move @10 ticks</div>';
      h += '<div class="mini-chart" style="height:120px"><canvas id="sigScatterC"></canvas></div>';
      h += '</div>';

      // Trader Leaderboard
      h += '<div class="chart-section">';
      h += '<div class="section-title">Trader Informativeness Leaderboard</div>';
      if (!sc.hasTraderNames) {
        h += '<div style="font-size:9px;color:#666;padding:4px">No trader names in data. Load output.log or local sandbox data to see per-trader hit rates.</div>';
      } else {
        const sorted = Object.entries(sc.traderStats).sort((a, b) => {
          const ra = a[1].hitRates[10]?.rate ?? 0;
          const rb = b[1].hitRates[10]?.rate ?? 0;
          return rb - ra;
        });
        h += '<div style="font-size:9px;overflow-x:auto">';
        h += '<table style="width:100%;border-collapse:collapse;font-size:9px">';
        h += '<tr style="color:#888"><th>Trader</th><th>Trades</th><th>AvgQty</th><th>Class</th>';
        for (const n of sc.lookaheads) h += `<th>@${n}</th>`;
        h += '</tr>';
        for (const [name, stats] of sorted) {
          const clsColor = stats.classification === 'informed' ? '#ffdd44' : '#666';
          h += `<tr><td style="color:${clsColor}">${name.substring(0, 12)}</td><td>${stats.count}</td><td>${stats.avgQty}</td><td style="color:${clsColor}">${stats.classification}</td>`;
          for (const n of sc.lookaheads) {
            const hr = stats.hitRates[n];
            if (!hr || hr.rate == null) { h += '<td>—</td>'; continue; }
            const cls = hr.rate > 55 ? 'good' : hr.rate < 45 ? 'bad' : 'val';
            h += `<td class="${cls}">${hr.rate}%<span style="color:#444;font-size:7px">(${hr.total})</span></td>`;
          }
          h += '</tr>';
        }
        h += '</table></div>';
      }
      h += '</div>';

      // Spike event list
      h += '<div class="chart-section">';
      h += '<div class="section-title">Spike Events (newest first, max 100)</div>';
      h += '<div style="max-height:200px;overflow-y:auto;font-size:9px">';
      h += '<table style="width:100%;border-collapse:collapse;font-size:9px">';
      h += '<tr style="color:#888"><th>Time</th><th>Side</th><th>Dir</th><th>Mag</th><th>Move@10</th></tr>';
      const displaySpikes = spikesWithOutcomes.slice(-100).reverse();
      for (const sp of displaySpikes) {
        const m10 = sp.outcomes[10];
        const m10Str = m10 != null ? m10.toFixed(2) : '—';
        const m10Cls = m10 != null ? (m10 > 0 ? 'good' : m10 < 0 ? 'bad' : 'val') : '';
        const sideColor = sp.side === 'bid' ? '#4488ff' : '#ff4444';
        h += `<tr class="spike-row" data-ts="${sp.ts}" style="cursor:pointer"><td>${sp.ts}</td><td style="color:${sideColor}">${sp.side}</td><td>${sp.direction === 'up' ? '&#9650;' : '&#9660;'}</td><td>${sp.magnitude.toFixed(1)}</td><td class="${m10Cls}">${m10Str}</td></tr>`;
      }
      h += '</table></div>';
      h += '</div>';

      h += '</div>';
      el.innerHTML = h;

      // Attach event handlers and render canvases
      setTimeout(() => {
        // Threshold slider
        const slider = document.getElementById('spikeThreshold');
        if (slider) {
          slider.addEventListener('input', e => {
            const thresh = parseFloat(e.target.value);
            document.getElementById('spikeThreshLabel').textContent = thresh.toFixed(1);
            const newSpikes = sc.detectSpikes(thresh);
            document.getElementById('spikeCount').textContent = newSpikes.length;
            // Re-render the full tab with new threshold
            const newWithOutcomes = sc.measureOutcomes(newSpikes, sc.lookaheads);
            renderSigTimeline(sc.snaps, newWithOutcomes);
            renderSigScatter(newWithOutcomes, sc.lookaheads);
          });
        }

        // Spike row click -> jump main chart
        document.querySelectorAll('.spike-row').forEach(row => {
          row.addEventListener('click', () => {
            const ts = parseFloat(row.dataset.ts);
            const range = vp.xMax - vp.xMin;
            vp.xMin = ts - range * 0.3;
            vp.xMax = ts + range * 0.7;
            scheduleRender();
          });
        });

        // Render canvases
        renderSigTimeline(sc.snaps, spikesWithOutcomes);
        renderSigScatter(spikesWithOutcomes, sc.lookaheads);
      }, 30);
    }

    function renderSigTimeline(snaps, spikes) {
      const c = document.getElementById('sigTimelineC'); if (!c) return;
      const { ctx, w, h } = setupCanvas(c, c.parentElement);
      if (!snaps.length) return;

      const tMin = snaps[0].ts, tMax = snaps[snaps.length - 1].ts;
      const tRange = tMax - tMin || 1;
      const fairs = snaps.filter(s => s.fair != null).map(s => s.fair);
      if (!fairs.length) return;
      const fMin = Math.min(...fairs) - 1, fMax = Math.max(...fairs) + 1;
      const fRange = fMax - fMin || 1;
      const tx = ts => ((ts - tMin) / tRange) * w;
      const ty = f => h - ((f - fMin) / fRange) * h;

      // Draw wall_mid line
      ctx.strokeStyle = '#44ff88';
      ctx.lineWidth = 0.5;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      let started = false;
      for (const s of snaps) {
        if (s.fair == null) continue;
        const x = tx(s.ts), y = ty(s.fair);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Draw spike markers
      for (const sp of spikes) {
        const x = tx(sp.ts);
        const y = ty(sp.wallMid || ((fMax + fMin) / 2));
        const size = Math.min(6, 2 + sp.magnitude);
        ctx.fillStyle = sp.direction === 'up' ? '#00ff88' : '#ff4444';
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        if (sp.direction === 'up') {
          ctx.moveTo(x, y - size); ctx.lineTo(x - size * 0.7, y + size * 0.5); ctx.lineTo(x + size * 0.7, y + size * 0.5);
        } else {
          ctx.moveTo(x, y + size); ctx.lineTo(x - size * 0.7, y - size * 0.5); ctx.lineTo(x + size * 0.7, y - size * 0.5);
        }
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Axis labels
      ctx.font = '7px Courier New';
      ctx.fillStyle = '#555';
      ctx.fillText(fMax.toFixed(0), 2, 9);
      ctx.fillText(fMin.toFixed(0), 2, h - 2);
      ctx.fillText(`${spikes.length} spikes`, w - 50, 9);
    }

    function renderSigScatter(spikes, lookaheads) {
      const c = document.getElementById('sigScatterC'); if (!c) return;
      const { ctx, w, h } = setupCanvas(c, c.parentElement);

      const n = 10; // lookahead for scatter
      const points = spikes.filter(s => s.outcomes[n] != null);
      if (!points.length) {
        ctx.fillStyle = '#555'; ctx.font = '10px Courier New';
        ctx.fillText('No data for scatter', 10, h / 2);
        return;
      }

      const mags = points.map(p => p.magnitude);
      const moves = points.map(p => p.outcomes[n]);
      const magMax = Math.max(...mags) * 1.1 || 1;
      const moveMax = Math.max(...moves.map(Math.abs)) * 1.1 || 1;
      const px = m => (m / magMax) * (w - 20) + 10;
      const py = m => h / 2 - (m / moveMax) * (h / 2 - 10);

      // Zero line
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

      // Points
      for (const p of points) {
        const x = px(p.magnitude);
        const y = py(p.outcomes[n]);
        ctx.fillStyle = p.direction === 'up' ? '#00ff88' : '#ff4444';
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Labels
      ctx.font = '7px Courier New';
      ctx.fillStyle = '#555';
      ctx.fillText('Magnitude →', w / 2 - 20, h - 2);
      ctx.fillText(`Move@${n} ↑`, 2, 9);
      ctx.fillText(`n=${points.length}`, w - 35, 9);
    }
