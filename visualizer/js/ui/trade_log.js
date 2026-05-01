// =====================================================================
    // SECTION 16 — TRADE LOG
    // =====================================================================
    function updateTradeLog() {
      const container = document.getElementById('tradeList');
      if (!container || !currentProduct) return;

      const sideFilter = document.getElementById('filterSide').value;
      const qtyMin = parseInt(document.getElementById('fQtyMin').value) || 0;
      const qtyMax = parseInt(document.getElementById('fQtyMax').value) || 9999;
      const distMin = parseFloat(document.getElementById('fDistMin').value) || 0;
      const traderFilter = document.getElementById('filterTrader')?.value || '';
      const snaps = getVisSnaps();

      const filtered = getVisTrades().filter(t => {
        if (t.qty < qtyMin || t.qty > qtyMax) return false;
        const isOwn = t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION';
        const isBuy = t.buyer === 'SUBMISSION' || (t.buyer && !isOwn);
        if (sideFilter === 'buy' && (!isBuy || isOwn)) return false;
        if (sideFilter === 'sell' && (isBuy || isOwn)) return false;
        if (sideFilter === 'own' && !isOwn) return false;
        if (traderFilter && t.buyer !== traderFilter && t.seller !== traderFilter) return false;
        if (distMin > 0) {
          const s = findNearest(snaps, t.ts);
          if (s && s.fair != null && Math.abs(t.price - s.fair) < distMin) return false;
        }
        const type = classifyTrade(t, findNearest(snaps, t.ts));
        if (!tradeTypeActive[type]) return false;
        return true;
      });

      const display = filtered.slice(-600);
      let html = '';
      for (const t of display) {
        const snap = findNearest(snaps, t.ts);
        const type = classifyTrade(t, snap);
        const edge = snap && snap.fair != null ? (t.price - snap.fair).toFixed(1) : '—';
        const isOwn = t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION';
        const side = t.buyer === 'SUBMISSION' ? 'B' : t.seller === 'SUBMISSION' ? 'S' : '?';
        html += `<div class="trade-row ${type}"><span>${t.ts}</span><span>${type}</span><span>${t.price}</span><span>${t.qty}</span><span>${edge}</span></div>`;
      }
      container.innerHTML = html;
      container.scrollTop = container.scrollHeight;
    }
