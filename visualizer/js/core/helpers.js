// =====================================================================
    // SECTION 5 — TRADE CLASSIFICATION
    // =====================================================================
    // Own fills: TB/TS (take buy/sell), MB/MS (make buy/sell), CB/CS (clear buy/sell)
    // Bot trades: M (inside spread), S (small ≤5), B (big >5)
    function classifyTrade(trade, snap) {
      const isOwn = trade.buyer === 'SUBMISSION' || trade.seller === 'SUBMISSION';
      if (isOwn) {
        const isBuy = trade.buyer === 'SUBMISSION';
        const sideChar = isBuy ? 'B' : 'S';

        // Match fill to our sent orders by price to determine take/make/clear
        if (snap && snap.sentOrders && snap.sentOrders.length) {
          const match = snap.sentOrders.find(o =>
            Math.abs(o.price - trade.price) < 0.01 &&
            ((o.side === 'buy') === isBuy)
          );
          if (match) {
            if (match.type === 'take') return 'T' + sideChar;
            if (match.type === 'make') return 'M' + sideChar;
            if (match.type === 'clear') return 'C' + sideChar;
          }
        }

        // Fallback: heuristic — compare fill price to fair value
        if (snap && snap.fair != null) {
          const edge = isBuy ? (snap.fair - trade.price) : (trade.price - snap.fair);
          if (Math.abs(edge) < 0.5) return 'C' + sideChar;  // near-zero edge → likely clear
          if (edge > 0) return 'T' + sideChar;  // positive edge → aggressive take
        }
        return 'M' + sideChar;  // default to make
      }

      // Bot/market trades
      if (snap && snap.bids.length && snap.asks.length) {
        const bestBid = Math.max(...snap.bids.map(b => b.price));
        const bestAsk = Math.min(...snap.asks.map(a => a.price));
        if (trade.price > bestBid && trade.price < bestAsk) return 'M';
      }

      return trade.qty <= SMALL_QTY_THRESHOLD ? 'S' : 'B';
    }

    function tradeBuyDir(trade, snap) {
      if (trade.buyer === 'SUBMISSION') return true;
      if (trade.seller === 'SUBMISSION') return false;
      if (!snap || !snap.asks.length) return true;
      const bestAsk = Math.min(...snap.asks.map(a => a.price));
      return trade.price >= bestAsk;
    }

    function isOwnFillType(type) {
      return type === 'TB' || type === 'TS' || type === 'MB' || type === 'MS' || type === 'CB' || type === 'CS';
    }

    // =====================================================================
    // SECTION 7 — DATA HELPERS
    // =====================================================================
    // Extract day number from IMC filename e.g. "prices_round_0_day_-1.csv" → -1
    function dayFromFilename(name) {
      const m = (name || '').match(/day_(-?\d+)/i);
      return m ? parseInt(m[1]) : null;
    }

    function getSnaps(product, day) {
      const arr = Store.data[product] || [];
      return day === 'all' ? arr : arr.filter(s => s.day === parseInt(day));
    }
    function getVisSnaps() { return getSnaps(currentProduct, currentDay); }
    function getRange() { const s = getVisSnaps(); return s.filter(s => s.ts >= vp.xMin && s.ts <= vp.xMax); }
    function downsample(arr, max) {
      if (arr.length <= max) return arr;
      const step = Math.ceil(arr.length / max);
      return arr.filter((_, i) => i % step === 0);
    }
    function findNearest(arr, ts) {
      if (!arr.length) return null;
      let lo = 0, hi = arr.length - 1;
      while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m].ts < ts) lo = m + 1; else hi = m; }
      if (lo > 0 && Math.abs(arr[lo - 1].ts - ts) < Math.abs(arr[lo].ts - ts)) lo--;
      return arr[lo];
    }
    function getVisTrades() {
      return Store.trades.filter(t => t.symbol === currentProduct && t.ts >= vp.xMin && t.ts <= vp.xMax);
    }
