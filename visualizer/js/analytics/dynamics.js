// =====================================================================
    // SECTION 21e — DYNAMICS ANALYSIS (OU, Roll, Regimes, OFI, Bot Fingerprints)
    // =====================================================================

    function fitOU(mids, dt) {
      // Ornstein-Uhlenbeck via OLS: dX = a + b*X, kappa=-b/dt, theta=-a/b, sigma=std(resid)/sqrt(dt)
      const n = mids.length - 1;
      if (n < 100) return { kappa: null, theta: null, sigma: null, halfLife: null, n, tStat: null, valid: false, reason: 'Need >=100 data points' };
      const dX = [], X = [];
      for (let i = 1; i <= n; i++) { dX.push(mids[i] - mids[i - 1]); X.push(mids[i - 1]); }
      let sx = 0, sy = 0, sxy = 0, sx2 = 0;
      for (let i = 0; i < n; i++) { sx += X[i]; sy += dX[i]; sxy += X[i] * dX[i]; sx2 += X[i] * X[i]; }
      const denom = n * sx2 - sx * sx;
      if (Math.abs(denom) < 1e-12) return { kappa: null, theta: null, sigma: null, halfLife: null, n, tStat: null, valid: false, reason: 'Degenerate data' };
      const b = (n * sxy - sx * sy) / denom;
      const a = (sy - b * sx) / n;
      const kappa = -b / dt;
      const theta = kappa !== 0 ? -a / b : mean(mids);
      const residuals = dX.map((d, i) => d - a - b * X[i]);
      const sigmaResid = std(residuals);
      const sigma = sigmaResid / Math.sqrt(dt);
      // Standard error of b
      const xm = sx / n;
      let ssx = 0; for (let i = 0; i < n; i++) ssx += (X[i] - xm) ** 2;
      const se_b = ssx > 0 ? sigmaResid / Math.sqrt(ssx) : Infinity;
      const tStat = se_b > 0 ? b / se_b : 0;
      return {
        kappa: +kappa.toFixed(6), theta: +theta.toFixed(2), sigma: +sigma.toFixed(4),
        halfLife: kappa > 0 ? +(Math.log(2) / kappa).toFixed(1) : null,
        n, tStat: +tStat.toFixed(2), valid: n >= 100 && kappa > 0
      };
    }

    function rollModel(returns) {
      // Roll (1984): implied_spread = 2*sqrt(max(0, -cov(r_t, r_{t-1})))
      if (returns.length < 50) return { autocovariance: null, impliedSpread: null, n: returns.length, valid: false };
      const mr = mean(returns);
      let cov = 0;
      for (let i = 1; i < returns.length; i++) cov += (returns[i] - mr) * (returns[i - 1] - mr);
      cov /= (returns.length - 1);
      const impliedSpread = cov < 0 ? 2 * Math.sqrt(-cov) : 0;
      return { autocovariance: +cov.toFixed(8), impliedSpread: +impliedSpread.toFixed(4), n: returns.length, valid: true };
    }

    function detectRegimes(mids, windowSize) {
      windowSize = windowSize || 50;
      if (mids.length < windowSize * 3) return { rollingVol: [], transitions: [], medianVol: null, valid: false };
      const changes = [];
      for (let i = 1; i < mids.length; i++) changes.push(mids[i] - mids[i - 1]);
      const rollingVol = [];
      for (let i = windowSize - 1; i < changes.length; i++) {
        const w = changes.slice(i - windowSize + 1, i + 1);
        rollingVol.push({ idx: i + 1, vol: std(w) }); // idx into mids array
      }
      const vols = rollingVol.map(r => r.vol);
      const medVol = med(vols);
      const quietThresh = medVol * 0.67, volThresh = medVol * 1.5;
      for (const r of rollingVol) {
        r.regime = r.vol > volThresh ? 'volatile' : r.vol < quietThresh ? 'quiet' : 'normal';
      }
      const transitions = [];
      for (let i = 1; i < rollingVol.length; i++) {
        if (rollingVol[i].regime !== rollingVol[i - 1].regime) {
          transitions.push({ idx: rollingVol[i].idx, from: rollingVol[i - 1].regime, to: rollingVol[i].regime, vol: rollingVol[i].vol });
        }
      }
      return { rollingVol, transitions, medianVol: +medVol.toFixed(4), valid: true };
    }

    function computeOFI(snaps) {
      // Order Flow Imbalance: delta(bid_vol_L1) - delta(ask_vol_L1) per tick
      if (snaps.length < 2) return { ofi: [], cumOFI: [], correlation: null, valid: false };
      const ofi = [], midChanges = [];
      for (let i = 1; i < snaps.length; i++) {
        const prev = snaps[i - 1], curr = snaps[i];
        const prevBV = prev.bids[0]?.vol || 0, currBV = curr.bids[0]?.vol || 0;
        const prevAV = prev.asks[0]?.vol || 0, currAV = curr.asks[0]?.vol || 0;
        const dBid = currBV - prevBV, dAsk = currAV - prevAV;
        ofi.push({ ts: curr.ts, value: dBid - dAsk });
        if (prev.mid != null && curr.mid != null) midChanges.push(curr.mid - prev.mid);
        else midChanges.push(0);
      }
      // Cumulative OFI
      let cum = 0;
      const cumOFI = ofi.map(o => { cum += o.value; return { ts: o.ts, value: cum }; });
      // Correlation between OFI and next-tick mid change
      let correlation = null;
      if (ofi.length > 100) {
        const ofiVals = ofi.map(o => o.value);
        const mofi = mean(ofiVals), mmc = mean(midChanges);
        let num = 0, d1 = 0, d2 = 0;
        for (let i = 0; i < ofiVals.length; i++) {
          const a = ofiVals[i] - mofi, b = midChanges[i] - mmc;
          num += a * b; d1 += a * a; d2 += b * b;
        }
        const denom = Math.sqrt(d1 * d2);
        correlation = denom > 0 ? +(num / denom).toFixed(4) : 0;
      }
      return { ofi, cumOFI, correlation, n: ofi.length, valid: ofi.length >= 50 };
    }

    function fingerprintBots(trades, snaps) {
      // Group market trades by quantity and price distance from mid
      const mkt = trades.filter(t => t.buyer !== 'SUBMISSION' && t.seller !== 'SUBMISSION');
      if (mkt.length < 50) return { archetypes: [], valid: false };
      const profiles = {};
      for (const t of mkt) {
        const snap = findNearest(snaps, t.ts);
        const distFromMid = snap && snap.mid != null ? +(t.price - snap.mid).toFixed(1) : null;
        const key = t.qty; // group by quantity
        if (!profiles[key]) profiles[key] = { qty: t.qty, count: 0, totalDist: 0, distN: 0, buys: 0, sells: 0 };
        profiles[key].count++;
        if (distFromMid != null) { profiles[key].totalDist += Math.abs(distFromMid); profiles[key].distN++; }
        const snap2 = findNearest(snaps, t.ts);
        if (snap2 && snap2.bids[0] && t.price >= snap2.bids[0].price) profiles[key].buys++;
        else profiles[key].sells++;
      }
      // Build archetype table sorted by frequency
      const archetypes = Object.values(profiles)
        .map(p => ({
          qty: p.qty, count: p.count,
          avgDistFromMid: p.distN > 0 ? +(p.totalDist / p.distN).toFixed(2) : null,
          buyPct: p.count > 0 ? +((p.buys / p.count) * 100).toFixed(0) : 50,
          label: p.qty <= 5 ? 'bot' : p.qty <= 15 ? 'medium' : 'large'
        }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 15);
      return { archetypes, totalMarketTrades: mkt.length, valid: true };
    }

    function computeDynamicsAnalysis(snaps, trades) {
      const mids = snaps.map(s => s.mid).filter(m => m != null);
      const returns = [];
      for (let i = 1; i < mids.length; i++) {
        if (mids[i] > 0 && mids[i - 1] > 0) returns.push(Math.log(mids[i] / mids[i - 1]));
      }
      const ou = fitOU(mids, 1);
      const roll = rollModel(returns);
      const regimes = detectRegimes(mids, 50);
      const ofi = computeOFI(snaps);
      const bots = fingerprintBots(trades, snaps);

      // Compute actual spread for Roll comparison
      const spreads = snaps.filter(s => s.bids.length && s.asks.length)
        .map(s => Math.min(...s.asks.map(a => a.price)) - Math.max(...s.bids.map(b => b.price)));
      const actualSpread = spreads.length ? +mean(spreads).toFixed(2) : null;

      // Generate actionable insights
      const insights = [];
      if (ou.valid && ou.kappa > 0.01 && Math.abs(ou.tStat) > 2) {
        const suggestedFactor = Math.min(0.5, ou.kappa * 0.8);
        insights.push({
          type: 'mean-reversion', confidence: 'high',
          message: `OU kappa=${ou.kappa} (half-life=${ou.halfLife} ticks) with strong significance (t=${ou.tStat}). Consider mean-reversion factor ${suggestedFactor.toFixed(3)}–${(suggestedFactor * 1.8).toFixed(3)}.`
        });
      } else if (ou.valid && ou.kappa > 0.01) {
        insights.push({
          type: 'mean-reversion', confidence: 'moderate',
          message: `OU kappa=${ou.kappa} (half-life=${ou.halfLife} ticks) but significance is weak (t=${ou.tStat}). Use cautiously.`
        });
      }
      if (roll.valid && roll.impliedSpread > 0 && actualSpread) {
        const ratio = (roll.impliedSpread / actualSpread).toFixed(2);
        if (+ratio > 1.3) {
          insights.push({ type: 'spread', confidence: 'moderate', message: `Roll implied spread (${roll.impliedSpread}) > actual (${actualSpread}) by ${ratio}x — bid-ask bounce is amplified, consider widening quotes.` });
        } else if (+ratio < 0.7) {
          insights.push({ type: 'spread', confidence: 'moderate', message: `Roll implied spread (${roll.impliedSpread}) < actual (${actualSpread}) — room to tighten quotes closer to fair.` });
        }
      }
      if (regimes.valid && regimes.transitions.length > 0) {
        const volRegimes = regimes.rollingVol.filter(r => r.regime === 'volatile');
        const volPct = ((volRegimes.length / regimes.rollingVol.length) * 100).toFixed(0);
        insights.push({ type: 'regime', confidence: 'high', message: `${regimes.transitions.length} regime transitions detected. Volatile ${volPct}% of the time. Widen make quotes by 1-2 ticks during volatile regimes.` });
      }
      if (ofi.valid && ofi.correlation != null && Math.abs(ofi.correlation) > 0.1) {
        insights.push({ type: 'flow', confidence: Math.abs(ofi.correlation) > 0.2 ? 'high' : 'moderate', message: `OFI→mid correlation=${ofi.correlation}. Order flow has predictive power — consider incorporating flow signal into fair value adjustment.` });
      }

      return { ou, roll, regimes, ofi, bots, actualSpread, insights, mids, returns };
    }

    function renderDynamicsTab() {
      const tab = document.getElementById('dynamicsTab');
      if (!currentProduct) { tab.innerHTML = '<div class="explain-box">Select a product first.</div>'; return; }
      const snaps = getVisSnaps();
      if (!snaps.length) { tab.innerHTML = '<div class="explain-box">No data loaded.</div>'; return; }

      if (!dynamicsCache) {
        const trades = Store.trades.filter(t => t.symbol === currentProduct);
        dynamicsCache = computeDynamicsAnalysis(snaps, trades);
      }
      const d = dynamicsCache;

      let html = '<div class="explain-box"><b>Dynamics Analysis:</b> Estimates underlying process parameters for this simulated market. OU=Ornstein-Uhlenbeck mean-reversion, Roll=implied spread from bid-ask bounce, Regimes=volatility clustering, OFI=order flow imbalance.</div>';

      // OU Parameters
      html += '<div style="margin:6px 0"><b style="color:#4cc9f0">Ornstein-Uhlenbeck Process</b></div>';
      if (d.ou.valid) {
        const tColor = Math.abs(d.ou.tStat) > 2 ? '#44ff44' : Math.abs(d.ou.tStat) > 1 ? '#ffaa00' : '#ff4444';
        html += `<table style="width:100%;font-size:10px;border-collapse:collapse">`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Kappa (mean-reversion speed)</td><td style="color:#fff"><b>${d.ou.kappa}</b></td></tr>`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Theta (long-term mean)</td><td style="color:#fff"><b>${d.ou.theta}</b></td></tr>`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Sigma (volatility)</td><td style="color:#fff"><b>${d.ou.sigma}</b></td></tr>`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Half-life (ticks)</td><td style="color:#fff"><b>${d.ou.halfLife ?? 'N/A'}</b></td></tr>`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">t-statistic</td><td style="color:${tColor}"><b>${d.ou.tStat}</b> ${Math.abs(d.ou.tStat) > 2 ? '(significant)' : Math.abs(d.ou.tStat) > 1 ? '(weak)' : '(not significant)'}</td></tr>`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Sample size</td><td style="color:#888">${d.ou.n}</td></tr>`;
        html += `</table>`;
      } else {
        html += `<div style="color:#ff6666;font-size:10px;padding:2px">${d.ou.reason || 'Insufficient data or non-mean-reverting'} (n=${d.ou.n})</div>`;
      }

      // Roll Model
      html += '<div style="margin:8px 0 4px"><b style="color:#4cc9f0">Roll Model (Bid-Ask Bounce)</b></div>';
      if (d.roll.valid) {
        html += `<table style="width:100%;font-size:10px;border-collapse:collapse">`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Implied spread</td><td style="color:#fff"><b>${d.roll.impliedSpread}</b></td></tr>`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Actual spread (avg)</td><td style="color:#fff"><b>${d.actualSpread ?? 'N/A'}</b></td></tr>`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Autocovariance (lag-1)</td><td style="color:#888">${d.roll.autocovariance}</td></tr>`;
        const ratio = d.actualSpread && d.roll.impliedSpread ? (d.roll.impliedSpread / d.actualSpread).toFixed(2) : 'N/A';
        html += `<tr><td style="color:#aaa;padding:2px 4px">Implied/Actual ratio</td><td style="color:${ratio > 1.3 ? '#ffaa00' : ratio < 0.7 ? '#44ff44' : '#fff'}">${ratio}</td></tr>`;
        html += `</table>`;
      } else {
        html += `<div style="color:#666;font-size:10px;padding:2px">Insufficient data (n=${d.roll.n}, need 50+)</div>`;
      }

      // OFI
      html += '<div style="margin:8px 0 4px"><b style="color:#4cc9f0">Order Flow Imbalance (OFI)</b></div>';
      if (d.ofi.valid) {
        const corrColor = d.ofi.correlation != null && Math.abs(d.ofi.correlation) > 0.1 ? '#44ff44' : '#888';
        html += `<table style="width:100%;font-size:10px;border-collapse:collapse">`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">OFI → mid-price correlation</td><td style="color:${corrColor}"><b>${d.ofi.correlation ?? 'N/A'}</b></td></tr>`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Sample size</td><td style="color:#888">${d.ofi.n}</td></tr>`;
        html += `</table>`;
        // OFI mini chart
        html += `<canvas id="ofiCanvas" width="340" height="100" style="width:100%;height:100px;margin:4px 0;background:#0a0e1a;border-radius:4px"></canvas>`;
      } else {
        html += `<div style="color:#666;font-size:10px;padding:2px">Insufficient data (n=${d.ofi.n})</div>`;
      }

      // Regime Detection
      html += '<div style="margin:8px 0 4px"><b style="color:#4cc9f0">Volatility Regimes</b></div>';
      if (d.regimes.valid) {
        html += `<table style="width:100%;font-size:10px;border-collapse:collapse">`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Median volatility</td><td style="color:#fff"><b>${d.regimes.medianVol}</b></td></tr>`;
        html += `<tr><td style="color:#aaa;padding:2px 4px">Regime transitions</td><td style="color:#fff"><b>${d.regimes.transitions.length}</b></td></tr>`;
        const rv = d.regimes.rollingVol;
        const quietPct = ((rv.filter(r => r.regime === 'quiet').length / rv.length) * 100).toFixed(0);
        const volPct = ((rv.filter(r => r.regime === 'volatile').length / rv.length) * 100).toFixed(0);
        html += `<tr><td style="color:#aaa;padding:2px 4px">Time in regimes</td><td><span style="color:#44ff44">${quietPct}% quiet</span> / <span style="color:#ff4444">${volPct}% volatile</span></td></tr>`;
        html += `</table>`;
        html += `<canvas id="regimeCanvas" width="340" height="80" style="width:100%;height:80px;margin:4px 0;background:#0a0e1a;border-radius:4px"></canvas>`;
      } else {
        html += `<div style="color:#666;font-size:10px;padding:2px">Insufficient data for regime detection (need 150+ snapshots)</div>`;
      }

      // Bot Fingerprints
      html += '<div style="margin:8px 0 4px"><b style="color:#4cc9f0">Bot Fingerprints</b></div>';
      if (d.bots.valid) {
        html += `<div style="font-size:9px;color:#888;margin-bottom:3px">${d.bots.totalMarketTrades} market trades analyzed</div>`;
        html += `<table style="width:100%;font-size:9px;border-collapse:collapse">`;
        html += `<tr style="color:#4cc9f0"><td style="padding:1px 3px">Qty</td><td>Count</td><td>AvgDist</td><td>Buy%</td><td>Type</td></tr>`;
        for (const a of d.bots.archetypes) {
          const typeColor = a.label === 'bot' ? '#88bbff' : a.label === 'medium' ? '#ffaa44' : '#ff4488';
          html += `<tr><td style="padding:1px 3px;color:#fff">${a.qty}</td><td style="color:#aaa">${a.count}</td><td style="color:#aaa">${a.avgDistFromMid ?? '—'}</td><td style="color:#aaa">${a.buyPct}%</td><td style="color:${typeColor}">${a.label}</td></tr>`;
        }
        html += `</table>`;
      } else {
        html += `<div style="color:#666;font-size:10px;padding:2px">Insufficient market trades (<50)</div>`;
      }

      // Actionable Insights
      if (d.insights.length) {
        html += '<div style="margin:10px 0 4px"><b style="color:#ffcc00">Actionable Insights</b></div>';
        for (const ins of d.insights) {
          const confColor = ins.confidence === 'high' ? '#44ff44' : ins.confidence === 'moderate' ? '#ffaa00' : '#666';
          const icon = ins.type === 'mean-reversion' ? '~' : ins.type === 'spread' ? '<>' : ins.type === 'regime' ? '#' : '>';
          html += `<div style="font-size:10px;padding:3px 4px;margin:2px 0;border-left:2px solid ${confColor};background:rgba(255,255,255,0.03)"><span style="color:${confColor}">[${icon}]</span> ${ins.message}</div>`;
        }
      }

      tab.innerHTML = html;

      // Render OFI canvas
      setTimeout(() => {
        const ofiC = document.getElementById('ofiCanvas');
        if (ofiC && d.ofi.valid && d.ofi.cumOFI.length > 1) {
          const ctx = ofiC.getContext('2d');
          const w = ofiC.width, h = ofiC.height;
          ctx.clearRect(0, 0, w, h);
          const data = d.ofi.cumOFI;
          const vals = data.map(d => d.value);
          const minV = Math.min(...vals), maxV = Math.max(...vals);
          const range = maxV - minV || 1;
          const tsMin = data[0].ts, tsMax = data[data.length - 1].ts;
          const tsRange = tsMax - tsMin || 1;
          // Zero line
          const y0 = h - ((-minV) / range) * h;
          ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5;
          ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(w, y0); ctx.stroke();
          // Cumulative OFI line
          ctx.strokeStyle = '#4cc9f0'; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.8;
          ctx.beginPath();
          for (let i = 0; i < data.length; i++) {
            const x = ((data[i].ts - tsMin) / tsRange) * w;
            const y = h - ((data[i].value - minV) / range) * h;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke(); ctx.globalAlpha = 1;
          // Labels
          ctx.fillStyle = '#4cc9f0'; ctx.font = '8px Courier New';
          ctx.fillText('Cumulative OFI', 4, 10);
          ctx.fillStyle = '#666';
          ctx.fillText(`${maxV.toFixed(0)}`, w - 30, 10);
          ctx.fillText(`${minV.toFixed(0)}`, w - 30, h - 3);
        }

        // Render regime canvas
        const regC = document.getElementById('regimeCanvas');
        if (regC && d.regimes.valid && d.regimes.rollingVol.length > 1) {
          const ctx = regC.getContext('2d');
          const w = regC.width, h = regC.height;
          ctx.clearRect(0, 0, w, h);
          const rv = d.regimes.rollingVol;
          const maxVol = Math.max(...rv.map(r => r.vol));
          const n = rv.length;
          const barW = Math.max(1, w / n);
          for (let i = 0; i < n; i++) {
            const r = rv[i];
            const x = (i / n) * w;
            const barH = (r.vol / (maxVol || 1)) * h;
            ctx.fillStyle = r.regime === 'volatile' ? 'rgba(255,68,68,0.6)' : r.regime === 'quiet' ? 'rgba(68,255,68,0.4)' : 'rgba(255,204,0,0.3)';
            ctx.fillRect(x, h - barH, barW + 0.5, barH);
          }
          // Transition markers
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.5;
          for (const t of d.regimes.transitions) {
            const x = ((t.idx - rv[0].idx) / (rv[rv.length - 1].idx - rv[0].idx || 1)) * w;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
          }
          // Labels
          ctx.fillStyle = '#888'; ctx.font = '8px Courier New';
          ctx.fillText('Rolling Volatility (50-tick)', 4, 10);
          ctx.fillStyle = '#44ff44'; ctx.fillText('quiet', w - 90, 10);
          ctx.fillStyle = '#ffcc00'; ctx.fillText('normal', w - 56, 10);
          ctx.fillStyle = '#ff4444'; ctx.fillText('volatile', w - 30, 10);
        }
      }, 30);
    }
