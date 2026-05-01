// SECTION 25 — DATA PARSING (prices)

function parsePricesCSV(text) {
      const lines = text.trim().split('\n');
      const startIdx = lines[0].startsWith('day;') ? 1 : 0;
      const daysSet = new Set(), productSet = new Set(), tmp = {};

      for (let i = startIdx; i < lines.length; i++) {
        const f = lines[i].split(';');
        if (f.length < 3) continue;
        const day = parseInt(f[0]), rawTs = parseInt(f[1]), product = f[2];
        if (isNaN(day) || isNaN(rawTs) || !product) continue;
        daysSet.add(day); productSet.add(product);

        const bids = [], asks = [];
        for (let j = 0; j < 3; j++) { const p = parseFloat(f[3 + j * 2]), v = parseInt(f[4 + j * 2]); if (!isNaN(p) && !isNaN(v)) bids.push({ price: p, vol: v }); }
        for (let j = 0; j < 3; j++) { const p = parseFloat(f[9 + j * 2]), v = parseInt(f[10 + j * 2]); if (!isNaN(p) && !isNaN(v)) asks.push({ price: p, vol: Math.abs(v) }); }

        if (!tmp[product]) tmp[product] = [];
        // Store rawTs — we'll normalize to localTs (0–999900) after computing dayOffsets
        // f[15]=mid_price, f[16]=profit_and_loss (f[14] is ask_volume_3, not mid)
        const midV = parseFloat(f[15]); const pnlV = parseFloat(f[16]);
        tmp[product].push({ day, rawTs, bids, asks, mid: isNaN(midV) ? null : midV, pnl: isNaN(pnlV) ? 0 : pnlV });
      }

      // Merge new days into Store.days and recompute ALL offsets so loading
      // files in any order produces correct global timestamps.
      const allDays = [...new Set([...Store.days, ...daysSet])].sort((a, b) => a - b);
      const minDay = allDays[0];
      allDays.forEach(d => { Store.dayOffsets[d] = (d - minDay) * 1000000; });
      Store.days = allDays;
      Store.products = [...new Set([...Store.products, ...productSet])].sort();

      // Re-index all previously loaded snapshots in case offsets shifted
      for (const product of Store.products) {
        for (const snap of (Store.data[product] || [])) {
          if (snap.localTs != null) snap.ts = (Store.dayOffsets[snap.day] || 0) + snap.localTs;
        }
        if (Store.data[product]) Store.data[product].sort((a, b) => a.ts - b.ts);
      }

      // Insert new snapshots (keyed by day+localTs to avoid duplicates)
      // output.log Activities section uses GLOBAL continuous timestamps (day=-1 ts=1000000-1999900),
      // while separate CSV files use LOCAL timestamps (each day resets to 0-999900).
      // Detect global timestamps: if rawTs > 999900, ts is already global → normalize to localTs.
      for (const product of productSet) {
        if (!Store.data[product]) Store.data[product] = [];
        const existing = new Map(Store.data[product].map(s => [s.day + '_' + s.localTs, s]));
        for (const row of (tmp[product] || [])) {
          const dayOffset = Store.dayOffsets[row.day] || 0;
          const isGlobal = row.rawTs > 999900;
          const ts = isGlobal ? row.rawTs : dayOffset + row.rawTs;
          const localTs = isGlobal ? row.rawTs - dayOffset : row.rawTs;
          const key = row.day + '_' + localTs;
          if (existing.has(key)) {
            const s = existing.get(key);
            if (!s.bids.length) {
              s.bids = row.bids;
              s.asks = row.asks;
              s.mid = row.mid;
              s.pnl = row.pnl;
            }
          } else {
            const snap = {
              ts, localTs, day: row.day,
              bids: row.bids, asks: row.asks,
              mid: row.mid, fair: null,
              pnl: row.pnl, pos: null,
              botOrders: [], ownTrades: [], logLines: [], sentOrders: [], marketTrades: []
            };
            Store.data[product].push(snap);
          }
        }
        Store.data[product].sort((a, b) => a.ts - b.ts);
        recomputeFairValuesForProduct(product);
      }
    }

    // Trades CSV — filename is used to infer the day and apply the correct global timestamp offset
    // (Trades CSVs have no day column, only raw timestamps 0-999900)
