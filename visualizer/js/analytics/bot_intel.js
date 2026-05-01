// =====================================================================
    // BOT INTEL — Exhaustive bot behavior & hidden taker analysis
    // =====================================================================

    // ── Computation helpers ──

    function buildTraderProfiles(mktTrades, ownTrades, snaps) {
      const profiles = {};
      const allTrades = [...mktTrades, ...ownTrades];

      for (const t of allTrades) {
        for (const name of [t.buyer, t.seller]) {
          if (!name || name === 'SUBMISSION') continue;
          if (!profiles[name]) profiles[name] = {
            name, trades: 0, totalQty: 0, buys: 0, sells: 0,
            distSum: 0, distN: 0, ticks: new Set(),
            firstTs: Infinity, lastTs: -Infinity,
            // 3-tick lookahead for informativeness
            correctDir: 0, dirN: 0,
            // counterparty to our makes
            ourMakeFills: 0,
          };
          const p = profiles[name];
          p.trades++;
          p.totalQty += t.qty;
          if (t.buyer === name) p.buys++;
          if (t.seller === name) p.sells++;
          p.ticks.add(t.ts);
          if (t.ts < p.firstTs) p.firstTs = t.ts;
          if (t.ts > p.lastTs) p.lastTs = t.ts;

          const snap = findNearest(snaps, t.ts);
          if (snap && snap.fair != null) {
            p.distSum += Math.abs(t.price - snap.fair);
            p.distN++;
          }

          // Informativeness: did FV move in trade direction 3 ticks later?
          if (snap && snap.fair != null) {
            const snapIdx = snaps.indexOf(snap);
            if (snapIdx >= 0 && snapIdx + 3 < snaps.length) {
              const futureFair = snaps[snapIdx + 3].fair;
              if (futureFair != null) {
                const isBuy = t.buyer === name;
                const moved = futureFair - snap.fair;
                if ((isBuy && moved > 0) || (!isBuy && moved < 0)) p.correctDir++;
                p.dirN++;
              }
            }
          }
        }
      }

      // Mark counterparty to our makes
      for (const t of ownTrades) {
        const snap = findNearest(snaps, t.ts);
        const type = classifyTrade(t, snap);
        if (type === 'MB' || type === 'MS') {
          const cp = t.buyer === 'SUBMISSION' ? t.seller : t.buyer;
          if (cp && profiles[cp]) profiles[cp].ourMakeFills++;
        }
      }

      const totalOwnMakes = ownTrades.filter(t => {
        const snap = findNearest(snaps, t.ts);
        const type = classifyTrade(t, snap);
        return type === 'MB' || type === 'MS';
      }).length;

      return Object.values(profiles).map(p => ({
        name: p.name,
        trades: p.trades,
        totalQty: p.totalQty,
        avgQty: +(p.totalQty / p.trades).toFixed(1),
        buyPct: +(p.buys / p.trades * 100).toFixed(0),
        avgDist: p.distN > 0 ? +(p.distSum / p.distN).toFixed(2) : null,
        activeTicks: p.ticks.size,
        firstTs: p.firstTs,
        lastTs: p.lastTs,
        informedPct: p.dirN > 10 ? +(p.correctDir / p.dirN * 100).toFixed(0) : null,
        ourMakeFills: p.ourMakeFills,
        ourMakePct: totalOwnMakes > 0 ? +(p.ourMakeFills / totalOwnMakes * 100).toFixed(1) : 0,
        classification: classifyTrader(p),
      })).sort((a, b) => b.trades - a.trades);
    }

    function classifyTrader(p) {
      const buyPct = p.buys / p.trades * 100;
      const avgDist = p.distN > 0 ? p.distSum / p.distN : 0;
      const informedPct = p.dirN > 10 ? p.correctDir / p.dirN * 100 : 50;
      if (informedPct > 60) return 'Informed';
      if (buyPct >= 40 && buyPct <= 60 && avgDist > 3) return 'MM';
      if (avgDist < 2) return 'Taker';
      return 'Bot';
    }

    function analyzeCounterparties(ownTrades, snaps) {
      const byType = { make: {}, take: {}, clear: {} };
      let totalMake = 0, totalTake = 0, totalClear = 0;

      for (const t of ownTrades) {
        const snap = findNearest(snaps, t.ts);
        const type = classifyTrade(t, snap);
        const isBuy = t.buyer === 'SUBMISSION';
        const cp = isBuy ? t.seller : t.buyer;
        if (!cp) continue;

        let bucket;
        if (type === 'MB' || type === 'MS') { bucket = 'make'; totalMake++; }
        else if (type === 'TB' || type === 'TS') { bucket = 'take'; totalTake++; }
        else { bucket = 'clear'; totalClear++; }

        if (!byType[bucket][cp]) byType[bucket][cp] = { name: cp, count: 0, qtySum: 0, edgeSum: 0, edgeN: 0 };
        const entry = byType[bucket][cp];
        entry.count++;
        entry.qtySum += t.qty;
        if (snap && snap.fair != null) {
          entry.edgeSum += (isBuy ? snap.fair - t.price : t.price - snap.fair);
          entry.edgeN++;
        }
      }

      const format = (bucket, total) => Object.values(bucket).map(e => ({
        name: e.name,
        count: e.count,
        pct: total > 0 ? +(e.count / total * 100).toFixed(1) : 0,
        avgQty: +(e.qtySum / e.count).toFixed(1),
        avgEdge: e.edgeN > 0 ? +(e.edgeSum / e.edgeN).toFixed(2) : null,
      })).sort((a, b) => b.count - a.count);

      return {
        make: format(byType.make, totalMake),
        take: format(byType.take, totalTake),
        clear: format(byType.clear, totalClear),
        totalMake, totalTake, totalClear,
      };
    }

    function computeFillRateByPrice(snaps) {
      const posted = {};  // offset → count of times posted
      const filled = {};  // offset → count of times filled

      for (const snap of snaps) {
        if (!snap.sentOrders || snap.fair == null) continue;

        const makeOrders = snap.sentOrders.filter(o => o.type === 'make');
        for (const o of makeOrders) {
          const offset = Math.round(o.price - snap.fair);
          const key = `${offset}_${o.side}`;
          posted[key] = (posted[key] || 0) + 1;
        }

        const makeFills = (snap.ownTrades || []).filter(t => {
          const type = classifyTrade(t, snap);
          return type === 'MB' || type === 'MS';
        });
        for (const t of makeFills) {
          const offset = Math.round(t.price - snap.fair);
          const isBuy = t.buyer === 'SUBMISSION';
          const key = `${offset}_${isBuy ? 'buy' : 'sell'}`;
          filled[key] = (filled[key] || 0) + 1;
        }
      }

      // Merge into array
      const allKeys = new Set([...Object.keys(posted), ...Object.keys(filled)]);
      const results = [];
      for (const key of allKeys) {
        const [offsetStr, side] = key.split('_');
        const offset = parseInt(offsetStr);
        const p = posted[key] || 0;
        const f = filled[key] || 0;
        results.push({
          offset, side, posted: p, filled: f,
          fillRate: p > 0 ? +(f / p * 100).toFixed(1) : 0,
        });
      }
      return results.sort((a, b) => a.offset - b.offset);
    }

    function analyzeQuoteOscillation(snaps) {
      const bestBids = [], bestAsks = [], l1Mids = [];
      const bidDeltas = [], askDeltas = [];

      for (const s of snaps) {
        const bb = s.bids.length ? Math.max(...s.bids.map(b => b.price)) : null;
        const ba = s.asks.length ? Math.min(...s.asks.map(a => a.price)) : null;
        bestBids.push(bb);
        bestAsks.push(ba);
        if (bb != null && ba != null) l1Mids.push((bb + ba) / 2);
        else l1Mids.push(null);
      }

      // Lag-1 autocorrelation
      const acBid = lag1AC(bestBids.filter(v => v != null));
      const acAsk = lag1AC(bestAsks.filter(v => v != null));
      const acMid = lag1AC(l1Mids.filter(v => v != null));

      // Tick-to-tick deltas
      const bidLevelFreq = {}, askLevelFreq = {};
      for (let i = 1; i < bestBids.length; i++) {
        if (bestBids[i] != null && bestBids[i - 1] != null) {
          const d = bestBids[i] - bestBids[i - 1];
          bidDeltas.push(d);
        }
        if (bestAsks[i] != null && bestAsks[i - 1] != null) {
          const d = bestAsks[i] - bestAsks[i - 1];
          askDeltas.push(d);
        }
        if (bestBids[i] != null) bidLevelFreq[bestBids[i]] = (bidLevelFreq[bestBids[i]] || 0) + 1;
        if (bestAsks[i] != null) askLevelFreq[bestAsks[i]] = (askLevelFreq[bestAsks[i]] || 0) + 1;
      }

      // Delta histogram
      const deltaHist = {};
      for (const d of bidDeltas) deltaHist[d] = (deltaHist[d] || 0) + 1;

      // Transition matrix (top 6 bid levels)
      const topBidLevels = Object.entries(bidLevelFreq).sort((a, b) => b[1] - a[1]).slice(0, 6).map(e => parseInt(e[0]));
      const transitions = {};
      for (let i = 1; i < bestBids.length; i++) {
        if (bestBids[i] != null && bestBids[i - 1] != null &&
            topBidLevels.includes(bestBids[i]) && topBidLevels.includes(bestBids[i - 1])) {
          const key = `${bestBids[i - 1]}→${bestBids[i]}`;
          transitions[key] = (transitions[key] || 0) + 1;
        }
      }

      // Last 200 two-sided ticks for chart
      const recentPairs = [];
      for (let i = Math.max(0, snaps.length - 400); i < snaps.length; i++) {
        if (bestBids[i] != null && bestAsks[i] != null) {
          recentPairs.push({ ts: snaps[i].ts, bid: bestBids[i], ask: bestAsks[i] });
        }
        if (recentPairs.length >= 200) break;
      }

      return {
        acBid: +acBid.toFixed(4),
        acAsk: +acAsk.toFixed(4),
        acMid: +acMid.toFixed(4),
        bidDeltaStd: bidDeltas.length ? +std(bidDeltas).toFixed(2) : null,
        askDeltaStd: askDeltas.length ? +std(askDeltas).toFixed(2) : null,
        bidDeltaMean: bidDeltas.length ? +mean(bidDeltas).toFixed(3) : null,
        deltaHist,
        topBidLevels: Object.entries(bidLevelFreq).sort((a, b) => b[1] - a[1]).slice(0, 8),
        topAskLevels: Object.entries(askLevelFreq).sort((a, b) => b[1] - a[1]).slice(0, 8),
        transitions,
        recentPairs,
      };
    }

    function lag1AC(arr) {
      if (arr.length < 3) return 0;
      const m = mean(arr);
      let num = 0, den = 0;
      for (let i = 0; i < arr.length; i++) {
        den += (arr[i] - m) * (arr[i] - m);
        if (i > 0) num += (arr[i] - m) * (arr[i - 1] - m);
      }
      return den === 0 ? 0 : num / den;
    }

    function analyzeBookRegimes(snaps) {
      const regimes = snaps.map(s => {
        const hasBids = s.bids && s.bids.length > 0;
        const hasAsks = s.asks && s.asks.length > 0;
        if (hasBids && hasAsks) return 'two-sided';
        if (hasBids) return 'bid-only';
        if (hasAsks) return 'ask-only';
        return 'empty';
      });

      // Counts
      const counts = { 'two-sided': 0, 'bid-only': 0, 'ask-only': 0, 'empty': 0 };
      for (const r of regimes) counts[r]++;
      const total = regimes.length;

      // Durations
      const durations = { 'two-sided': [], 'bid-only': [], 'ask-only': [], 'empty': [] };
      let runLen = 1;
      for (let i = 1; i < regimes.length; i++) {
        if (regimes[i] === regimes[i - 1]) { runLen++; }
        else { durations[regimes[i - 1]].push(runLen); runLen = 1; }
      }
      durations[regimes[regimes.length - 1]].push(runLen);

      // Transition matrix
      const trans = {};
      const types = ['two-sided', 'bid-only', 'ask-only', 'empty'];
      for (const a of types) for (const b of types) trans[`${a}→${b}`] = 0;
      for (let i = 1; i < regimes.length; i++) trans[`${regimes[i - 1]}→${regimes[i]}`]++;

      // Post-transition FV movement
      const postTransFV = [];
      for (let i = 1; i < regimes.length; i++) {
        if (regimes[i] !== regimes[i - 1] && snaps[i].fair != null) {
          const fv0 = snaps[i].fair;
          const fv3 = (i + 3 < snaps.length && snaps[i + 3].fair != null) ? snaps[i + 3].fair : null;
          const fv5 = (i + 5 < snaps.length && snaps[i + 5].fair != null) ? snaps[i + 5].fair : null;
          postTransFV.push({
            from: regimes[i - 1], to: regimes[i],
            dFV3: fv3 != null ? fv3 - fv0 : null,
            dFV5: fv5 != null ? fv5 - fv0 : null,
          });
        }
      }

      // Regime timeline for chart (downsample to 500 points)
      const step = Math.max(1, Math.floor(regimes.length / 500));
      const timeline = [];
      for (let i = 0; i < regimes.length; i += step) {
        timeline.push({ ts: snaps[i].ts, regime: regimes[i] });
      }

      return {
        counts, total, types,
        avgDuration: Object.fromEntries(types.map(t => [t, durations[t].length ? +mean(durations[t]).toFixed(1) : 0])),
        trans, postTransFV, timeline,
      };
    }

    function analyzeTradeTiming(trades, snaps) {
      const sorted = [...trades].sort((a, b) => a.ts - b.ts);
      const intervals = [];
      for (let i = 1; i < sorted.length; i++) intervals.push(sorted[i].ts - sorted[i - 1].ts);

      // Trades per tick
      const perTick = {};
      for (const t of sorted) perTick[t.ts] = (perTick[t.ts] || 0) + 1;
      const tickCounts = Object.values(perTick);
      const maxBurst = tickCounts.length ? Math.max(...tickCounts) : 0;

      // Per-trader intervals
      const byTrader = {};
      for (const t of sorted) {
        for (const name of [t.buyer, t.seller]) {
          if (!name || name === 'SUBMISSION') continue;
          if (!byTrader[name]) byTrader[name] = [];
          byTrader[name].push(t.ts);
        }
      }
      const traderIntervals = {};
      for (const [name, tsList] of Object.entries(byTrader)) {
        if (tsList.length < 2) continue;
        const tsSorted = [...tsList].sort((a, b) => a - b);
        const ints = [];
        for (let i = 1; i < tsSorted.length; i++) ints.push(tsSorted[i] - tsSorted[i - 1]);
        traderIntervals[name] = {
          mean: +mean(ints).toFixed(0),
          median: +med(ints).toFixed(0),
          std: +std(ints).toFixed(0),
          count: tsSorted.length,
          regularity: std(ints) > 0 ? +(mean(ints) / std(ints)).toFixed(2) : 999,
        };
      }

      // Rolling trade intensity (50-tick window) for chart
      const intensity = [];
      if (snaps.length > 0) {
        const tsMin = snaps[0].ts, tsMax = snaps[snaps.length - 1].ts;
        const step = Math.max(100, Math.floor((tsMax - tsMin) / 300));
        for (let ts = tsMin; ts <= tsMax; ts += step) {
          const count = sorted.filter(t => t.ts >= ts - 2500 && t.ts < ts + 2500).length;
          intensity.push({ ts, count });
        }
      }

      return {
        totalTrades: sorted.length,
        meanInterval: intervals.length ? +mean(intervals).toFixed(0) : null,
        medianInterval: intervals.length ? +med(intervals).toFixed(0) : null,
        avgPerTick: tickCounts.length ? +mean(tickCounts).toFixed(2) : null,
        maxBurst,
        traderIntervals: Object.entries(traderIntervals)
          .map(([name, s]) => ({ name, ...s }))
          .sort((a, b) => b.count - a.count),
        intensity,
      };
    }

    function generateInsights(intel) {
      const insights = [];

      // Hidden taker detection
      if (intel.counterparty.make.length > 0) {
        const top = intel.counterparty.make[0];
        if (top.pct > 40) {
          insights.push({ type: 'opportunity', text: `Hidden taker detected: <b>${top.name}</b> fills ${top.pct}% of your make orders (${top.count}/${intel.counterparty.totalMake}). This bot aggressively crosses your quotes. Consider targeting their preferred price level.` });
        }
        if (top.pct > 20 && top.pct <= 40) {
          insights.push({ type: 'info', text: `Top counterparty: <b>${top.name}</b> fills ${top.pct}% of your makes. Not dominant enough to be a hidden taker, but worth monitoring.` });
        }
      }

      // Fill rate sweet spots
      const goodFills = intel.fillRateByPrice.filter(f => f.fillRate > 40 && f.posted >= 20);
      if (goodFills.length) {
        const best = goodFills.sort((a, b) => b.fillRate - a.fillRate)[0];
        insights.push({ type: 'opportunity', text: `Fill rate sweet spot: FV${best.offset >= 0 ? '+' : ''}${best.offset} (${best.side}) has <b>${best.fillRate}%</b> fill rate (${best.filled}/${best.posted} ticks). Consider fixed-offset quoting at this level.` });
      }

      // Quote oscillation
      if (intel.quoteOscillation.acBid < -0.3) {
        insights.push({ type: 'opportunity', text: `Strong bid oscillation detected (lag-1 AC = <b>${intel.quoteOscillation.acBid}</b>). The bot's best bid alternates predictably. Fixed-offset quoting will outperform book-following.` });
      }
      if (intel.quoteOscillation.acAsk < -0.3) {
        insights.push({ type: 'opportunity', text: `Strong ask oscillation detected (lag-1 AC = <b>${intel.quoteOscillation.acAsk}</b>). The bot's best ask alternates predictably.` });
      }

      // One-sided book
      const oneSidedPct = ((intel.bookRegimes.counts['bid-only'] + intel.bookRegimes.counts['ask-only']) / intel.bookRegimes.total * 100);
      if (oneSidedPct > 10) {
        insights.push({ type: 'caution', text: `One-sided book on <b>${oneSidedPct.toFixed(1)}%</b> of ticks. FV model must hold steady through these gaps. Fixed-offset quoting handles this naturally.` });
      }

      // Informed traders
      for (const p of intel.traderProfiles) {
        if (p.informedPct != null && p.informedPct > 60 && p.trades > 20) {
          insights.push({ type: 'opportunity', text: `Trader <b>${p.name}</b> appears informed: FV moves in their trade direction <b>${p.informedPct}%</b> of the time (${p.trades} trades). Consider following their signal.` });
        }
      }

      // Regular traders (systematic bots)
      for (const t of intel.tradeTiming.traderIntervals) {
        if (t.regularity > 3 && t.count > 50) {
          insights.push({ type: 'info', text: `Trader <b>${t.name}</b> trades very regularly (mean interval ${t.mean}, CV=${(1/t.regularity).toFixed(2)}). Likely a systematic bot — their timing is predictable.` });
        }
      }

      if (insights.length === 0) {
        insights.push({ type: 'info', text: 'No strong signals detected. Load more data or try a different product.' });
      }

      return insights;
    }

    // ── Main computation ──

    function computeBotIntel(snaps, trades) {
      const mkt = trades.filter(t => t.buyer !== 'SUBMISSION' && t.seller !== 'SUBMISSION');
      const own = trades.filter(t => t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION');

      return {
        traderProfiles: buildTraderProfiles(mkt, own, snaps),
        counterparty: analyzeCounterparties(own, snaps),
        fillRateByPrice: computeFillRateByPrice(snaps),
        quoteOscillation: analyzeQuoteOscillation(snaps),
        bookRegimes: analyzeBookRegimes(snaps),
        tradeTiming: analyzeTradeTiming(trades, snaps),
        insights: [],
      };
    }

    // ── Rendering ──

    function renderBotIntelTab() {
      const el = document.getElementById('botIntelTab');
      const snaps = getVisSnaps();
      const trades = Store.trades.filter(t => t.symbol === currentProduct);

      if (snaps.length < 50) {
        el.innerHTML = '<div style="color:#555">Need 50+ snapshots for Bot Intel analysis</div>';
        return;
      }

      if (!botIntelCache) {
        el.innerHTML = '<div style="color:#4cc9f0">Computing bot intelligence...</div>';
        setTimeout(() => {
          botIntelCache = computeBotIntel(snaps, trades);
          botIntelCache.insights = generateInsights(botIntelCache);
          renderBotIntelInner(el);
        }, 30);
      } else {
        renderBotIntelInner(el);
      }
    }

    function renderBotIntelInner(el) {
      const c = botIntelCache;
      let h = '';

      // ── Section 7: Insights (top of tab for immediate visibility) ──
      h += '<div class="section-title" style="color:#e94560">Actionable Insights</div>';
      for (const ins of c.insights) {
        const color = ins.type === 'opportunity' ? '#44ff88' : ins.type === 'caution' ? '#ffaa00' : '#4cc9f0';
        const icon = ins.type === 'opportunity' ? '+' : ins.type === 'caution' ? '!' : 'i';
        h += `<div style="font-size:9px;margin:3px 0;padding:3px 6px;border-left:3px solid ${color};background:${color}11"><span style="color:${color};font-weight:bold">[${icon}]</span> ${ins.text}</div>`;
      }

      // ── Section 1: Trader Profiles ──
      h += '<div class="section-title">Trader Profiles</div>';
      if (c.traderProfiles.length) {
        h += '<div style="overflow-x:auto"><table style="font-size:8px;border-collapse:collapse;width:100%">';
        h += '<tr style="color:#888;border-bottom:1px solid #333"><th style="text-align:left;padding:2px 4px">Name</th><th>Trades</th><th>Qty</th><th>Buy%</th><th>AvgDist</th><th>AvgQty</th><th>Ticks</th><th>Dir%</th><th>OurMake%</th><th>Class</th></tr>';
        for (const p of c.traderProfiles.slice(0, 20)) {
          const classColor = p.classification === 'Informed' ? '#ffdd44' : p.classification === 'MM' ? '#4cc9f0' : p.classification === 'Taker' ? '#ff4488' : '#aaa';
          const highlightBg = p.ourMakePct > 30 ? 'background:rgba(255,68,68,0.15)' : '';
          h += `<tr style="border-bottom:1px solid #222;${highlightBg}">`;
          h += `<td style="padding:2px 4px;color:#ddd">${p.name}</td>`;
          h += `<td style="text-align:center">${p.trades}</td>`;
          h += `<td style="text-align:center">${p.totalQty}</td>`;
          h += `<td style="text-align:center;color:${p.buyPct > 60 ? '#44ff88' : p.buyPct < 40 ? '#ff4488' : '#aaa'}">${p.buyPct}%</td>`;
          h += `<td style="text-align:center">${p.avgDist ?? '—'}</td>`;
          h += `<td style="text-align:center">${p.avgQty}</td>`;
          h += `<td style="text-align:center">${p.activeTicks}</td>`;
          h += `<td style="text-align:center;color:${p.informedPct != null && p.informedPct > 55 ? '#ffdd44' : '#888'}">${p.informedPct != null ? p.informedPct + '%' : '—'}</td>`;
          h += `<td style="text-align:center;color:${p.ourMakePct > 30 ? '#ff4488' : '#888'}">${p.ourMakePct > 0 ? p.ourMakePct + '%' : '—'}</td>`;
          h += `<td style="text-align:center;color:${classColor}">${p.classification}</td>`;
          h += '</tr>';
        }
        h += '</table></div>';
        h += '<div style="font-size:8px;color:#555;margin:2px 0">Dir% = FV moves in trade direction within 3 ticks. OurMake% = % of your passive fills from this trader. Red highlight = potential hidden taker.</div>';
      } else {
        h += '<div style="font-size:9px;color:#555">No trader names available. Load output.log with trade history for counterparty data.</div>';
      }

      // ── Section 2: Counterparty Analysis ──
      h += '<div class="section-title">Counterparty Analysis (Who Fills Your Orders?)</div>';
      const cpMake = c.counterparty.make;
      if (cpMake.length) {
        h += `<div style="font-size:9px;color:#aaa;margin:2px 0">Your <b>make</b> fills (${c.counterparty.totalMake} total) — counterparty breakdown:</div>`;
        h += '<div style="overflow-x:auto"><table style="font-size:8px;border-collapse:collapse;width:100%">';
        h += '<tr style="color:#888;border-bottom:1px solid #333"><th style="text-align:left;padding:2px 4px">Counterparty</th><th>Fills</th><th>Share</th><th>AvgQty</th><th>AvgEdge</th></tr>';
        for (const cp of cpMake.slice(0, 10)) {
          const bg = cp.pct > 40 ? 'background:rgba(255,68,68,0.2)' : cp.pct > 20 ? 'background:rgba(255,170,0,0.1)' : '';
          h += `<tr style="border-bottom:1px solid #222;${bg}"><td style="padding:2px 4px">${cp.name}</td><td style="text-align:center">${cp.count}</td><td style="text-align:center;color:${cp.pct > 40 ? '#ff4488' : '#aaa'}">${cp.pct}%</td><td style="text-align:center">${cp.avgQty}</td><td style="text-align:center;color:${(cp.avgEdge ?? 0) > 0 ? '#44ff88' : '#ff4488'}">${cp.avgEdge ?? '—'}</td></tr>`;
        }
        h += '</table></div>';
      } else {
        h += '<div style="font-size:9px;color:#555">No make fills with counterparty data. Ensure output.log has trade history with trader names.</div>';
      }

      const cpTake = c.counterparty.take;
      if (cpTake.length) {
        h += `<div style="font-size:9px;color:#aaa;margin:4px 0 2px">Your <b>take</b> fills (${c.counterparty.totalTake} total):</div>`;
        h += '<div style="overflow-x:auto"><table style="font-size:8px;border-collapse:collapse;width:100%">';
        h += '<tr style="color:#888;border-bottom:1px solid #333"><th style="text-align:left;padding:2px 4px">Counterparty</th><th>Fills</th><th>Share</th><th>AvgEdge</th></tr>';
        for (const cp of cpTake.slice(0, 10)) {
          h += `<tr style="border-bottom:1px solid #222"><td style="padding:2px 4px">${cp.name}</td><td style="text-align:center">${cp.count}</td><td style="text-align:center">${cp.pct}%</td><td style="text-align:center;color:${(cp.avgEdge ?? 0) > 0 ? '#44ff88' : '#ff4488'}">${cp.avgEdge ?? '—'}</td></tr>`;
        }
        h += '</table></div>';
      }

      // ── Section 3: Fill Rate by Price Level ──
      h += '<div class="section-title">Fill Rate by Price Offset from FV</div>';
      if (c.fillRateByPrice.length) {
        h += '<div class="mini-chart" style="height:130px"><canvas id="fillRateCanvas"></canvas></div>';
        h += '<div style="overflow-x:auto"><table style="font-size:8px;border-collapse:collapse;width:100%">';
        h += '<tr style="color:#888;border-bottom:1px solid #333"><th style="text-align:left;padding:2px 4px">Offset</th><th>Side</th><th>Posted</th><th>Filled</th><th>Fill%</th></tr>';
        for (const f of c.fillRateByPrice) {
          if (f.posted < 5) continue;
          const rateColor = f.fillRate > 50 ? '#44ff88' : f.fillRate > 20 ? '#ffaa00' : '#ff4444';
          h += `<tr style="border-bottom:1px solid #222"><td style="padding:2px 4px">${f.offset >= 0 ? '+' : ''}${f.offset}</td><td style="text-align:center">${f.side}</td><td style="text-align:center">${f.posted}</td><td style="text-align:center">${f.filled}</td><td style="text-align:center;color:${rateColor}"><b>${f.fillRate}%</b></td></tr>`;
        }
        h += '</table></div>';
      } else {
        h += '<div style="font-size:9px;color:#555">No sentOrders data. Ensure output.log has ORD| structured logging lines.</div>';
      }

      // ── Section 4: Quote Oscillation ──
      h += '<div class="section-title">Quote Oscillation Detector</div>';
      const osc = c.quoteOscillation;
      h += '<div class="stat-grid">';
      h += `<div class="lbl">Bid lag-1 AC</div><div class="val" style="color:${osc.acBid < -0.3 ? '#ff4488' : '#aaa'}">${osc.acBid}</div>`;
      h += `<div class="lbl">Ask lag-1 AC</div><div class="val" style="color:${osc.acAsk < -0.3 ? '#ff4488' : '#aaa'}">${osc.acAsk}</div>`;
      h += `<div class="lbl">L1 Mid lag-1 AC</div><div class="val" style="color:${osc.acMid < -0.3 ? '#ff4488' : '#aaa'}">${osc.acMid}</div>`;
      h += `<div class="lbl">Bid delta std</div><div class="val">${osc.bidDeltaStd ?? '—'}</div>`;
      h += `<div class="lbl">Ask delta std</div><div class="val">${osc.askDeltaStd ?? '—'}</div>`;
      h += '</div>';

      // Top bid/ask levels
      h += '<div style="font-size:8px;color:#888;margin:4px 0 1px">Most common bid levels:</div>';
      h += '<div style="font-size:8px">';
      for (const [level, cnt] of osc.topBidLevels.slice(0, 6)) {
        h += `<span style="color:#4488ff">${level}</span><span style="color:#555">×${cnt}</span> `;
      }
      h += '</div>';
      h += '<div style="font-size:8px;color:#888;margin:2px 0 1px">Most common ask levels:</div>';
      h += '<div style="font-size:8px">';
      for (const [level, cnt] of osc.topAskLevels.slice(0, 6)) {
        h += `<span style="color:#ff4444">${level}</span><span style="color:#555">×${cnt}</span> `;
      }
      h += '</div>';

      // Transition matrix
      if (Object.keys(osc.transitions).length > 0) {
        h += '<div style="font-size:8px;color:#888;margin:4px 0 1px">Bid transition matrix (top levels):</div>';
        h += '<div style="font-size:8px">';
        const sorted = Object.entries(osc.transitions).sort((a, b) => b[1] - a[1]);
        for (const [key, cnt] of sorted.slice(0, 12)) {
          h += `<span style="color:#4cc9f0">${key}</span>: ${cnt}  `;
        }
        h += '</div>';
      }

      // Delta histogram & sawtooth chart
      h += '<div style="display:flex;gap:4px">';
      h += '<div class="mini-chart" style="height:90px;flex:1"><canvas id="deltaHistCanvas"></canvas></div>';
      h += '<div class="mini-chart" style="height:90px;flex:2"><canvas id="sawtoothCanvas"></canvas></div>';
      h += '</div>';
      h += '<div style="font-size:7px;color:#555">Left: bid tick-to-tick delta distribution. Right: best bid (blue) / ask (red) over last 200 two-sided ticks.</div>';

      // ── Section 5: Book Regimes ──
      h += '<div class="section-title">Book Regime Analysis</div>';
      const reg = c.bookRegimes;
      h += '<div class="stat-grid">';
      for (const t of reg.types) {
        const pct = (reg.counts[t] / reg.total * 100).toFixed(1);
        const col = t === 'two-sided' ? '#44ff88' : t === 'bid-only' ? '#4488ff' : t === 'ask-only' ? '#ff4444' : '#555';
        h += `<div class="lbl">${t}</div><div class="val" style="color:${col}">${pct}% (avg ${reg.avgDuration[t]} ticks)</div>`;
      }
      h += '</div>';

      // Transition matrix
      h += '<div style="font-size:8px;color:#888;margin:4px 0 1px">Regime transitions (count):</div>';
      h += '<div style="overflow-x:auto"><table style="font-size:7px;border-collapse:collapse">';
      h += '<tr><th style="padding:1px 3px"></th>';
      for (const t of reg.types) h += `<th style="padding:1px 3px;color:#888">${t.slice(0, 3)}</th>`;
      h += '</tr>';
      for (const from of reg.types) {
        h += `<tr><td style="padding:1px 3px;color:#888">${from.slice(0, 3)}</td>`;
        for (const to of reg.types) {
          const cnt = reg.trans[`${from}→${to}`] || 0;
          const bg = cnt > 0 ? `rgba(76,201,240,${Math.min(0.4, cnt / reg.total * 5)})` : 'transparent';
          h += `<td style="padding:1px 3px;text-align:center;background:${bg}">${cnt || ''}</td>`;
        }
        h += '</tr>';
      }
      h += '</table></div>';

      // Regime timeline
      h += '<div class="mini-chart" style="height:25px"><canvas id="regimeCanvas"></canvas></div>';
      h += '<div style="font-size:7px;color:#555">Green=two-sided, Blue=bid-only, Red=ask-only, Gray=empty</div>';

      // ── Section 6: Trade Timing ──
      h += '<div class="section-title">Trade Timing & Clustering</div>';
      const tt = c.tradeTiming;
      h += '<div class="stat-grid">';
      h += `<div class="lbl">Total trades</div><div class="val">${tt.totalTrades}</div>`;
      h += `<div class="lbl">Mean interval</div><div class="val">${tt.meanInterval ?? '—'}</div>`;
      h += `<div class="lbl">Median interval</div><div class="val">${tt.medianInterval ?? '—'}</div>`;
      h += `<div class="lbl">Avg trades/tick</div><div class="val">${tt.avgPerTick ?? '—'}</div>`;
      h += `<div class="lbl">Max burst</div><div class="val">${tt.maxBurst}</div>`;
      h += '</div>';

      if (tt.traderIntervals.length) {
        h += '<div style="font-size:8px;color:#888;margin:4px 0 1px">Per-trader timing (regularity = mean/std, higher = more periodic):</div>';
        h += '<div style="overflow-x:auto"><table style="font-size:8px;border-collapse:collapse;width:100%">';
        h += '<tr style="color:#888;border-bottom:1px solid #333"><th style="text-align:left;padding:2px 4px">Trader</th><th>Trades</th><th>MeanInt</th><th>MedInt</th><th>StdInt</th><th>Regularity</th></tr>';
        for (const t of tt.traderIntervals.slice(0, 12)) {
          const regColor = t.regularity > 3 ? '#ffdd44' : '#888';
          h += `<tr style="border-bottom:1px solid #222"><td style="padding:2px 4px">${t.name}</td><td style="text-align:center">${t.count}</td><td style="text-align:center">${t.mean}</td><td style="text-align:center">${t.median}</td><td style="text-align:center">${t.std}</td><td style="text-align:center;color:${regColor}">${t.regularity}</td></tr>`;
        }
        h += '</table></div>';
      }

      h += '<div class="mini-chart" style="height:70px"><canvas id="intensityCanvas"></canvas></div>';
      h += '<div style="font-size:7px;color:#555">Rolling trade intensity (5-second window)</div>';

      el.innerHTML = h;

      // ── Canvas rendering (deferred) ──
      setTimeout(() => renderBotIntelCanvases(c), 30);
    }

    function renderBotIntelCanvases(c) {
      // 1. Fill rate bar chart
      const frc = document.getElementById('fillRateCanvas');
      if (frc && c.fillRateByPrice.length) {
        const { ctx, w, h } = setupCanvas(frc, frc.parentElement);
        const data = c.fillRateByPrice.filter(f => f.posted >= 5);
        if (data.length) {
          const offsets = data.map(d => d.offset);
          const minO = Math.min(...offsets), maxO = Math.max(...offsets);
          const range = maxO - minO || 1;
          const barW = Math.max(4, (w - 40) / (range + 1));
          const xOff = 30;

          // Y axis (0-100%)
          ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5;
          ctx.font = '7px Courier New'; ctx.fillStyle = '#555';
          for (let pct = 0; pct <= 100; pct += 25) {
            const y = h - 15 - (pct / 100) * (h - 25);
            ctx.beginPath(); ctx.moveTo(xOff, y); ctx.lineTo(w, y); ctx.stroke();
            ctx.fillText(pct + '%', 2, y + 3);
          }

          for (const d of data) {
            const x = xOff + (d.offset - minO) / range * (w - xOff - 10);
            const barH = (d.fillRate / 100) * (h - 25);
            const color = d.fillRate > 50 ? 'rgba(68,255,136,0.7)' : d.fillRate > 20 ? 'rgba(255,170,0,0.7)' : 'rgba(255,68,68,0.5)';
            ctx.fillStyle = color;
            ctx.fillRect(x - barW / 2, h - 15 - barH, barW - 1, barH);

            // Labels
            ctx.fillStyle = '#666'; ctx.font = '6px Courier New'; ctx.textAlign = 'center';
            ctx.fillText((d.offset >= 0 ? '+' : '') + d.offset, x, h - 4);
            if (d.fillRate > 0) {
              ctx.fillStyle = '#ddd'; ctx.font = '7px Courier New';
              ctx.fillText(d.fillRate.toFixed(0) + '%', x, h - 18 - barH);
            }
          }
          ctx.textAlign = 'start';
        }
      }

      // 2. Delta histogram
      const dhc = document.getElementById('deltaHistCanvas');
      if (dhc && Object.keys(c.quoteOscillation.deltaHist).length) {
        const { ctx, w, h } = setupCanvas(dhc, dhc.parentElement);
        const hist = c.quoteOscillation.deltaHist;
        const deltas = Object.keys(hist).map(Number).sort((a, b) => a - b);
        const counts = deltas.map(d => hist[d]);
        const maxC = Math.max(...counts);
        const barW = Math.max(6, (w - 10) / deltas.length);

        for (let i = 0; i < deltas.length; i++) {
          const x = 5 + i * barW;
          const barH = (counts[i] / maxC) * (h - 18);
          const color = deltas[i] === 0 ? 'rgba(76,201,240,0.6)' : deltas[i] > 0 ? 'rgba(68,255,136,0.5)' : 'rgba(255,68,136,0.5)';
          ctx.fillStyle = color;
          ctx.fillRect(x, h - 12 - barH, barW - 1, barH);
          ctx.fillStyle = '#666'; ctx.font = '6px Courier New'; ctx.textAlign = 'center';
          ctx.fillText(deltas[i].toString(), x + barW / 2, h - 2);
        }
        ctx.textAlign = 'start';
      }

      // 3. Sawtooth chart (best bid/ask last 200 ticks)
      const stc = document.getElementById('sawtoothCanvas');
      if (stc && c.quoteOscillation.recentPairs.length > 2) {
        const { ctx, w, h } = setupCanvas(stc, stc.parentElement);
        const pairs = c.quoteOscillation.recentPairs;
        const allPrices = pairs.flatMap(p => [p.bid, p.ask]);
        const pMin = Math.min(...allPrices), pMax = Math.max(...allPrices);
        const pRange = pMax - pMin || 1;
        const toX = i => (i / (pairs.length - 1)) * w;
        const toY = p => h - 5 - ((p - pMin) / pRange) * (h - 10);

        // Grid
        ctx.strokeStyle = '#222'; ctx.lineWidth = 0.5;
        for (let p = Math.ceil(pMin); p <= pMax; p++) {
          const y = toY(p);
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }

        // Bid line
        ctx.strokeStyle = '#4488ff'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < pairs.length; i++) {
          const x = toX(i), y = toY(pairs[i].bid);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Ask line
        ctx.strokeStyle = '#ff4444'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < pairs.length; i++) {
          const x = toX(i), y = toY(pairs[i].ask);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#555'; ctx.font = '7px Courier New';
        ctx.fillText(pMin.toFixed(0), 2, h - 2);
        ctx.textAlign = 'right'; ctx.fillText(pMax.toFixed(0), w - 2, 10);
        ctx.textAlign = 'start';
      }

      // 4. Regime timeline
      const rtc = document.getElementById('regimeCanvas');
      if (rtc && c.bookRegimes.timeline.length) {
        const { ctx, w, h } = setupCanvas(rtc, rtc.parentElement);
        const tl = c.bookRegimes.timeline;
        const colors = { 'two-sided': '#44ff88', 'bid-only': '#4488ff', 'ask-only': '#ff4444', 'empty': '#444' };
        const segW = w / tl.length;
        for (let i = 0; i < tl.length; i++) {
          ctx.fillStyle = colors[tl[i].regime] || '#444';
          ctx.fillRect(i * segW, 0, segW + 1, h);
        }
      }

      // 5. Trade intensity
      const tic = document.getElementById('intensityCanvas');
      if (tic && c.tradeTiming.intensity.length > 2) {
        const { ctx, w, h } = setupCanvas(tic, tic.parentElement);
        const data = c.tradeTiming.intensity;
        const maxC = Math.max(...data.map(d => d.count), 1);
        const toX = i => (i / (data.length - 1)) * w;
        const toY = v => h - 5 - (v / maxC) * (h - 10);

        ctx.fillStyle = 'rgba(76,201,240,0.15)';
        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let i = 0; i < data.length; i++) ctx.lineTo(toX(i), toY(data[i].count));
        ctx.lineTo(w, h);
        ctx.fill();

        ctx.strokeStyle = '#4cc9f0'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
          const x = toX(i), y = toY(data[i].count);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = '#555'; ctx.font = '7px Courier New';
        ctx.fillText('0', 2, h - 2); ctx.fillText(maxC.toString(), 2, 10);
      }
    }
