// =====================================================================
    // SECTION 22 — EXPORT ANALYSIS
    // =====================================================================
    function exportAnalysis() {
      if (!currentProduct) { alert('Select a product first'); return; }
      const snaps = getVisSnaps();
      if (!snaps.length) { alert('No data to export'); return; }

      const spreads = snaps.filter(s => s.bids.length && s.asks.length)
        .map(s => Math.min(...s.asks.map(a => a.price)) - Math.max(...s.bids.map(b => b.price)));

      const mids = snaps.map(s => s.mid).filter(m => m != null && m > 0);
      const returns = [];
      for (let i = 1; i < mids.length; i++) returns.push(Math.log(mids[i] / mids[i - 1]));

      const maxLag = Math.min(50, Math.floor(returns.length / 5));
      const mr = mean(returns), vr = vari(returns);
      const confBand = 1.96 / Math.sqrt(Math.max(returns.length, 1));
      const autocorr = [];
      for (let lag = 1; lag <= maxLag; lag++) {
        let s = 0, n = 0;
        for (let i = lag; i < returns.length; i++) { s += (returns[i] - mr) * (returns[i - lag] - mr); n++; }
        autocorr.push({ lag, ac: n && vr ? +(s / (n * vr)).toFixed(6) : 0 });
      }
      const sigLags = autocorr.filter(a => Math.abs(a.ac) > confBand);

      const trades = Store.trades.filter(t => t.symbol === currentProduct);
      const own = trades.filter(t => t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION');
      const mkt = trades.filter(t => t.buyer !== 'SUBMISSION' && t.seller !== 'SUBMISSION');
      const ownEdges = own.map(t => { const s = findNearest(snaps, t.ts); return s && s.fair != null ? +(t.price - s.fair).toFixed(2) : null; }).filter(e => e != null);

      const positions = snaps.map(s => s.pos).filter(p => p != null);
      const pnls = snaps.map(s => s.pnl).filter(p => p != null);
      let maxPnl = 0, maxDD = 0;
      for (const p of pnls) { if (p > maxPnl) maxPnl = p; const dd = maxPnl - p; if (dd > maxDD) maxDD = dd; }

      const qtyFreq = {};
      for (const t of trades) { qtyFreq[t.qty] = (qtyFreq[t.qty] || 0) + 1; }
      const topQtys = Object.entries(qtyFreq).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([q, c]) => ({ qty: +q, count: c }));

      const dailyStats = {};
      for (const s of snaps) { if (s.mid == null) continue; if (!dailyStats[s.day]) dailyStats[s.day] = { min: Infinity, max: -Infinity }; if (s.mid < dailyStats[s.day].min) dailyStats[s.day].min = s.mid; if (s.mid > dailyStats[s.day].max) dailyStats[s.day].max = s.mid; }
      for (const d of Object.values(dailyStats)) { if (d.min === Infinity) d.min = null; if (d.max === -Infinity) d.max = null; }

      const extremeTrades = [];
      for (const t of trades) {
        for (const [day, st] of Object.entries(dailyStats)) {
          if (st.min && Math.abs(t.price - st.min) <= 3) { extremeTrades.push({ ts: t.ts, price: t.price, qty: t.qty, type: 'nearLow', day: +day }); break; }
          if (st.max && Math.abs(t.price - st.max) <= 3) { extremeTrades.push({ ts: t.ts, price: t.price, qty: t.qty, type: 'nearHigh', day: +day }); break; }
        }
      }

      const out = {
        metadata: {
          product: currentProduct,
          fairValueMethod: PRODUCT_CONFIG[currentProduct]?.fairValue != null ? 'fixed' : resolveFairValueMethod(currentProduct),
          positionLimit: PRODUCT_CONFIG[currentProduct]?.posLimit || 80,
          days: Store.days,
          totalSnapshots: snaps.length,
          totalTrades: trades.length,
          timestampRange: [snaps[0].ts, snaps[snaps.length - 1].ts],
          fairValueNote: getFairValueMethodNote(currentProduct),
          fairValueParams: { emaAlpha: fvEmaAlpha }
        },
        spreadAnalysis: {
          mean: spreads.length ? +mean(spreads).toFixed(3) : null,
          median: spreads.length ? +med(spreads).toFixed(3) : null,
          stdDev: spreads.length ? +std(spreads).toFixed(3) : null,
          min: spreads.length ? +Math.min(...spreads).toFixed(1) : null,
          max: spreads.length ? +Math.max(...spreads).toFixed(1) : null,
          p10: spreads.length ? +pctile(spreads, 10).toFixed(2) : null,
          p90: spreads.length ? +pctile(spreads, 90).toFixed(2) : null
        },
        returnAnalysis: {
          mean: +mean(returns).toExponential(4),
          stdDev: +std(returns).toExponential(4),
          skewness: +skew(returns).toFixed(4),
          kurtosis: +kurt(returns).toFixed(4),
          n: returns.length,
          significanceThreshold: +confBand.toFixed(6),
          autocorrelation: autocorr,
          significantLags: sigLags,
          interpretation: sigLags.length ? `Lag-1 AC=${sigLags.find(a => a.lag === 1)?.ac || 'N/A'}: ${(sigLags.find(a => a.lag === 1)?.ac || 0) < 0 ? 'MEAN-REVERTING (negative lag-1)' : 'TRENDING (positive lag-1)'}. Use mean-reversion strategy if lag-1 is significantly negative.` : 'No significant autocorrelation — close to random walk, focus on market-making edge not directional trading.'
        },
        tradeStats: {
          own: { count: own.length, buys: own.filter(t => t.buyer === 'SUBMISSION').length, sells: own.filter(t => t.seller === 'SUBMISSION').length, totalQty: own.reduce((s, t) => s + t.qty, 0), avgEdge: ownEdges.length ? +mean(ownEdges).toFixed(3) : null, medianEdge: ownEdges.length ? +med(ownEdges).toFixed(3) : null, edgeStdDev: ownEdges.length ? +std(ownEdges).toFixed(3) : null, pctProfitable: ownEdges.length ? +(ownEdges.filter(e => e > 0).length / ownEdges.length * 100).toFixed(1) : null },
          market: { count: mkt.length, avgQty: mkt.length ? +mean(mkt.map(t => t.qty)).toFixed(1) : null, topQuantities: topQtys, extremeTrades: extremeTrades.slice(0, 20) }
        },
        positionStats: {
          max: positions.length ? Math.max(...positions) : 0,
          min: positions.length ? Math.min(...positions) : 0,
          avgAbs: positions.length ? +mean(positions.map(Math.abs)).toFixed(1) : 0,
          timeAtLimitPct: positions.length ? +(positions.filter(p => Math.abs(p) >= (PRODUCT_CONFIG[currentProduct]?.posLimit || 80) * 0.9).length / positions.length * 100).toFixed(1) : 0
        },
        pnlSummary: {
          final: pnls.length ? pnls[pnls.length - 1] : 0,
          max: +maxPnl.toFixed(0),
          maxDrawdown: +maxDD.toFixed(0)
        },
        dailyRanges: dailyStats,
        strategyInsights: {
          recommendedAction: sigLags.find(a => a.lag === 1 && a.ac < -0.05) ? 'Consider adding mean-reversion signal to trading — significant negative lag-1 autocorrelation detected.' : spreads.length && mean(spreads) > 4 ? 'Wide spreads — consider tightening make orders for more fills.' : 'Focus on market-making edge. No strong directional signal.',
          fairValueNote: getFairValueMethodNote(currentProduct)
        }
      };

      // Enhanced strategy analysis sections
      const sa = computeStrategyAnalysis(snaps, trades);
      out.fillAnalysis = {
        buysBelowFVPct: sa.fillAnalysis.buysBelowFVPct,
        sellsAboveFVPct: sa.fillAnalysis.sellsAboveFVPct,
        avgBuyEdge: sa.fillAnalysis.avgBuyEdge,
        avgSellEdge: sa.fillAnalysis.avgSellEdge,
        adverseSelectionScore: sa.fillAnalysis.adverseSelectionScore,
        edgeOverTime: [...sa.fillAnalysis.buyEdges.map(e => ({ ts: e.ts, edge: e.edge, side: 'buy' })),
        ...sa.fillAnalysis.sellEdges.map(e => ({ ts: e.ts, edge: e.edge, side: 'sell' }))]
          .sort((a, b) => a.ts - b.ts).slice(0, 200)
      };
      out.positionPatterns = {
        zeroCrossings: sa.positionPatterns.zeroCrossings,
        zeroCrossingsPerDay: sa.positionPatterns.zeroCrossingsPerDay,
        sawtoothScore: sa.positionPatterns.sawtoothScore,
        avgAbsDrift: sa.positionPatterns.avgAbsDrift,
        driftDirection: sa.positionPatterns.driftDirection,
        timeAtLimit: sa.positionPatterns.timeAtLimit
      };
      out.pnlSlope = {
        overallPerTick: sa.pnlSlope.overallPerTick,
        steadyPeriods: sa.pnlSlope.steadyPeriods,
        flatPeriods: sa.pnlSlope.flatPeriods,
        adversePeriods: sa.pnlSlope.adversePeriods,
        maxDrawdownDuration: sa.pnlSlope.maxDrawdownDuration,
        recoveryTime: sa.pnlSlope.recoveryTime
      };
      out.spreadDangerZones = {
        tightPeriods: sa.spreadAnalysis.tightZones.slice(0, 20),
        widePeriods: sa.spreadAnalysis.wideZones.slice(0, 20),
        tightSpreadFillRate: sa.spreadAnalysis.tightSpreadFillRate
      };
      out.tradeQtySignatures = {
        distribution: sa.tradeQtySignatures.distribution.slice(0, 15),
        botCluster: sa.tradeQtySignatures.botCluster,
        informedSizes: sa.tradeQtySignatures.informedSizes,
        singleTradeCount: sa.tradeQtySignatures.singleTradeCount
      };

      // === ENHANCED EXPORT SECTIONS (dynamics + portfolio) ===

      // Order book profile
      out.orderBookProfile = (() => {
        const depthByLevel = [[], [], []];
        let hasL3 = 0, totalSnaps = 0;
        for (const s of snaps) {
          totalSnaps++;
          for (let lvl = 0; lvl < 3; lvl++) {
            if (s.bids[lvl]) depthByLevel[lvl].push(s.bids[lvl].vol);
            if (s.asks[lvl]) depthByLevel[lvl].push(s.asks[lvl].vol);
          }
          if (s.bids[2] || s.asks[2]) hasL3++;
        }
        const bidVol1 = snaps.map(s => s.bids[0]?.vol || 0);
        const askVol1 = snaps.map(s => s.asks[0]?.vol || 0);
        const imbalance = bidVol1.map((bv, i) => {
          const total = bv + askVol1[i];
          return total > 0 ? (bv - askVol1[i]) / total : 0;
        });
        return {
          avgVolByLevel: depthByLevel.map(a => a.length ? +mean(a).toFixed(1) : null),
          l3AvailabilityPct: totalSnaps ? +(hasL3 / totalSnaps * 100).toFixed(1) : 0,
          depthImbalance: { mean: +mean(imbalance).toFixed(4), std: +std(imbalance).toFixed(4) }
        };
      })();

      // OU process parameters
      out.ouParameters = (() => {
        const m = snaps.map(s => s.mid).filter(v => v != null);
        if (m.length < 100) return { valid: false, reason: 'insufficient data', n: m.length };
        return fitOU(m, 1);
      })();

      // Regime analysis
      out.regimeAnalysis = (() => {
        const m = snaps.map(s => s.mid).filter(v => v != null);
        const reg = detectRegimes(m, 50);
        if (!reg.valid) return { valid: false };
        const rv = reg.rollingVol;
        return {
          valid: true, medianVol: reg.medianVol,
          transitions: reg.transitions.slice(0, 20),
          quietPct: +((rv.filter(r => r.regime === 'quiet').length / rv.length) * 100).toFixed(1),
          volatilePct: +((rv.filter(r => r.regime === 'volatile').length / rv.length) * 100).toFixed(1)
        };
      })();

      // Order flow imbalance
      out.orderFlowImbalance = (() => {
        const ofiResult = computeOFI(snaps);
        if (!ofiResult.valid) return { valid: false, n: ofiResult.n };
        return { valid: true, correlation: ofiResult.correlation, n: ofiResult.n,
          finalCumOFI: ofiResult.cumOFI.length ? ofiResult.cumOFI[ofiResult.cumOFI.length - 1].value : 0 };
      })();

      // Round-trip analysis
      out.roundTripAnalysis = (() => {
        const ownSorted = own.slice().sort((a, b) => a.ts - b.ts);
        const trips = [];
        let inventory = 0, costBasis = 0;
        for (const t of ownSorted) {
          const isBuy = t.buyer === 'SUBMISSION';
          if (isBuy) {
            costBasis += t.price * t.qty;
            inventory += t.qty;
          } else {
            if (inventory > 0) {
              const avgEntry = costBasis / inventory;
              const exitQty = Math.min(t.qty, inventory);
              const tripPnl = (t.price - avgEntry) * exitQty;
              trips.push({ exitTs: t.ts, pnl: +tripPnl.toFixed(2), qty: exitQty, entryAvg: +avgEntry.toFixed(2), exitPrice: t.price });
              inventory -= exitQty;
              costBasis = inventory > 0 ? (costBasis / (inventory + exitQty)) * inventory : 0;
            }
          }
        }
        return {
          totalRoundTrips: trips.length,
          avgPnlPerTrip: trips.length ? +mean(trips.map(t => t.pnl)).toFixed(2) : null,
          winRate: trips.length ? +((trips.filter(t => t.pnl > 0).length / trips.length) * 100).toFixed(1) : null,
          trips: trips.slice(0, 100)
        };
      })();

      // GraphLog total portfolio PnL
      out.graphLogPnl = Store.graphLog && Store.graphLog.length ? {
        dataPoints: Store.graphLog.length,
        final: Store.graphLog[Store.graphLog.length - 1].value,
        max: Math.max(...Store.graphLog.map(g => g.value)),
        min: Math.min(...Store.graphLog.map(g => g.value))
      } : null;

      // Per-product PnL comparison
      out.perProductPnlComparison = (() => {
        const result = {};
        for (const prod of Store.products) {
          const ps = (Store.data[prod] || []).filter(s => s.pnl != null);
          if (!ps.length) continue;
          result[prod] = { finalPnl: ps[ps.length - 1].pnl, maxPnl: Math.max(...ps.map(s => s.pnl)), snapshots: ps.length };
        }
        return Object.keys(result).length ? result : null;
      })();

      // Edge decay over time
      out.edgeDecay = (() => {
        if (ownEdges.length < 20) return null;
        const q = Math.floor(ownEdges.length / 4);
        return {
          q1AvgEdge: +mean(ownEdges.slice(0, q)).toFixed(3),
          q2AvgEdge: +mean(ownEdges.slice(q, q * 2)).toFixed(3),
          q3AvgEdge: +mean(ownEdges.slice(q * 2, q * 3)).toFixed(3),
          q4AvgEdge: +mean(ownEdges.slice(-q)).toFixed(3),
          trend: Math.abs(mean(ownEdges.slice(-q))) < Math.abs(mean(ownEdges.slice(0, q))) * 0.7 ? 'decaying' : 'stable_or_improving'
        };
      })();

      // Inventory metrics
      out.inventoryMetrics = (() => {
        if (!positions.length) return null;
        const absPos = positions.map(Math.abs);
        return {
          meanAbsInventory: +mean(absPos).toFixed(1),
          maxAbsInventory: Math.max(...absPos),
          inventoryCostProxy: +(mean(absPos) * (spreads.length ? mean(spreads) : 0) / 2).toFixed(1)
        };
      })();

      // Bot fingerprints
      out.botFingerprints = (() => {
        const bots = fingerprintBots(trades, snaps);
        if (!bots.valid) return null;
        return { totalMarketTrades: bots.totalMarketTrades, archetypes: bots.archetypes };
      })();

      document.getElementById('exportText').value = JSON.stringify(out, null, 2);
      document.getElementById('exportModal').classList.add('show');
    }
