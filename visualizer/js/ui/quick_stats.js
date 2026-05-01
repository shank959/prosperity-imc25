// =====================================================================
    // SECTION 17 — QUICK STATS
    // =====================================================================
    function updateQuickStats() {
      const el = document.getElementById('quickStats');
      if (!currentProduct) { el.innerHTML = ''; return; }
      const snaps = getVisSnaps();
      const trades = Store.trades.filter(t => t.symbol === currentProduct);
      const own = trades.filter(t => t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION');
      const last = snaps[snaps.length - 1];
      const pnl = last ? last.pnl : 0;
      const spreads = snaps.filter(s => s.bids.length && s.asks.length)
        .map(s => Math.min(...s.asks.map(a => a.price)) - Math.max(...s.bids.map(b => b.price)));
      const avgSpr = spreads.length ? mean(spreads).toFixed(1) : '—';
      const ownSnaps = own.map(t => { const s = findNearest(snaps, t.ts); return s && s.fair != null ? Math.abs(t.price - s.fair) : null; }).filter(e => e != null);
      const avgEdge = ownSnaps.length ? mean(ownSnaps).toFixed(2) : '—';
      // Adverse selection & sawtooth quick stats
      let advSel = '—', sawtooth = '—';
      if (snaps.length > 50) {
        const buyEdges = [], sellEdges = [];
        for (const t of own) {
          const s = findNearest(snaps, t.ts);
          if (!s || s.fair == null) continue;
          const edge = t.price - s.fair;
          if (t.buyer === 'SUBMISSION') buyEdges.push(edge); else sellEdges.push(edge);
        }
        const advB = buyEdges.length ? buyEdges.filter(e => e > 0).length / buyEdges.length : 0;
        const advS = sellEdges.length ? sellEdges.filter(e => e < 0).length / sellEdges.length : 0;
        advSel = ((advB + advS) / 2 * 100).toFixed(0);

        const positions = snaps.map(s => s.pos).filter(p => p != null);
        let zx = 0;
        for (let i = 1; i < positions.length; i++) {
          if ((positions[i - 1] > 0 && positions[i] <= 0) || (positions[i - 1] < 0 && positions[i] >= 0)) zx++;
        }
        const stRaw = positions.length > 100 ? zx / (positions.length / 1000) : 0;
        sawtooth = Math.min(100, stRaw * 10).toFixed(0);
      }
      const imcBadge = Store.sourceType === 'official' ? `<span class="official-badge">IMC</span>` : '';
      const totalPnlStr = Store.totalProfit != null ? ` | <span title="Total profit from IMC official JSON">TotPnL: <span class="val" style="color:${Store.totalProfit > 0 ? '#44ff44' : '#ff4444'}">${Store.totalProfit.toFixed(0)}</span></span>` : '';
      el.innerHTML = `${imcBadge}PnL: <span class="val">${pnl != null ? pnl.toFixed(0) : '—'}</span>${totalPnlStr} | Fills: <span class="val">${own.length}</span> | AvgEdge: <span class="val">${avgEdge}</span> | AvgSprd: <span class="val">${avgSpr}</span> | Adv%: <span class="val" style="color:${+advSel > 40 ? '#ff4444' : '#44ff44'}">${advSel}</span> | Saw: <span class="val" style="color:${+sawtooth > 30 ? '#44ff44' : '#ff4444'}">${sawtooth}</span>`;
    }
