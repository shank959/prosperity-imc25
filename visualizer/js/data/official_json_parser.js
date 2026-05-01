// SECTION 25 — DATA PARSING (official json)

function parseIMCOfficialJSON(text) {
      const obj = JSON.parse(text);
      Store.sourceType = 'official';

      // 1. OB snapshots — same CSV format as output.log activities section
      if (obj.activitiesLog) parsePricesCSV(obj.activitiesLog);

      // 2. Trade history — timestamps are GLOBAL (same coordinate system as activitiesLog)
      if (Array.isArray(obj.tradeHistory)) {
        for (const t of obj.tradeHistory) {
          Store.trades.push({
            ts: t.timestamp, buyer: t.buyer || '', seller: t.seller || '',
            symbol: t.symbol, price: t.price, qty: t.quantity
          });
        }
      }

      // 3. PnL curve — LOCAL timestamps (0–999900), detect day boundaries via rollback
      if (obj.graphLog) {
        const sortedDays = [...Store.days].sort((a, b) => a - b);
        const lines = obj.graphLog.trim().split('\n');
        const si = lines[0].startsWith('timestamp') ? 1 : 0;
        Store.graphLog = [];
        let dayIdx = 0, prevTs = -1;
        for (let i = si; i < lines.length; i++) {
          const parts = lines[i].split(';');
          if (parts.length < 2) continue;
          const localTs = parseInt(parts[0]), value = parseFloat(parts[1]);
          if (isNaN(localTs) || isNaN(value)) continue;
          // Detect day boundary: timestamp rolls back to a lower value
          if (localTs < prevTs && dayIdx < sortedDays.length - 1) dayIdx++;
          prevTs = localTs;
          const day = sortedDays[dayIdx] ?? sortedDays[0];
          const globalTs = (Store.dayOffsets[day] || 0) + localTs;
          Store.graphLog.push({ ts: globalTs, value });
        }
      }

      // 4. Metadata
      if (Array.isArray(obj.positions)) {
        Store.finalPositions = {};
        for (const p of obj.positions) Store.finalPositions[p.symbol] = p.quantity;
      }
      if (obj.profit != null) Store.totalProfit = obj.profit;

      // 5. Enrich snapshots with print lines + parse ORD|/MT| into sentOrders + FV/OB structured data
      if (Array.isArray(obj.logs) && obj.logs.length) {
        const sortedDays = [...Store.days].sort((a, b) => a - b);
        const orderRe = /^\[(MAKE|CLEAR)\s+(BID|OFFER)\]\s+(\d+)\s+(\w+)\s+@\s+(\d+)/;
        const ordRe = /^ORD\|(\w+)\|(TAKE|MAKE|CLEAR)\|(BUY|SELL)\|([\d.]+)\|(\d+)/;
        const mtRe = /^MT\|(\w+)\|([\d.]+)\|(\d+)\|([^|]*)\|([^|]*)/;
        const fvRe = /^FV\|(\w+)\|([\d.]+)\|POS\|(-?\d+)\|CAP\|(\d+)/;
        const obRe = /^OB\|(\w+)\|(\d+)LVL\|BVOL=(\d+)\|AVOL=(\d+)\|SPR=(-?[\d.]+)/;
        const logMap = new Map();

        // Detect day boundaries via timestamp rollback (same as parseSandboxLogs)
        let dayIdx = 0, prevLocalTs = -1;
        for (const entry of obj.logs) {
          const localTs = entry.timestamp;
          if (localTs < prevLocalTs && dayIdx < sortedDays.length - 1) dayIdx++;
          prevLocalTs = localTs;

          const day = sortedDays[dayIdx] ?? sortedDays[0];
          const globalTs = (Store.dayOffsets[day] || 0) + localTs;

          const allLines = (entry.lambdaLog || '').split('\n').filter(l => l.trim());
          const printLines = [];
          const ordersByProduct = {};
          const ordByProduct = {};
          const mtByProduct = {};
          const fvByProduct = {};
          const obByProduct = {};

          for (const line of allLines) {
            // Legacy [MAKE/CLEAR] format for botOrders rendering
            const m = line.match(orderRe);
            if (m) {
              const action = m[1], direction = m[2], qty = parseInt(m[3]), product = m[4], price = parseInt(m[5]);
              if (!ordersByProduct[product]) ordersByProduct[product] = [];
              ordersByProduct[product].push({ price, qty: direction === 'OFFER' ? -qty : qty, type: action.toLowerCase() });
            }
            // New ORD| format for fill classification + botOrders
            const ordm = line.match(ordRe);
            if (ordm) {
              const prod = ordm[1], type = ordm[2].toLowerCase(), side = ordm[3].toLowerCase();
              const price = parseFloat(ordm[4]), qty = parseInt(ordm[5]);
              if (!ordByProduct[prod]) ordByProduct[prod] = [];
              ordByProduct[prod].push({ type, side, price, qty });
              // Also populate botOrders for make/clear quote rendering
              if (type !== 'take') {
                if (!ordersByProduct[prod]) ordersByProduct[prod] = [];
                ordersByProduct[prod].push({ price, qty: side === 'sell' ? -qty : qty, type });
              }
            }
            const mtm = line.match(mtRe);
            if (mtm) {
              const prod = mtm[1];
              if (!mtByProduct[prod]) mtByProduct[prod] = [];
              mtByProduct[prod].push({ price: parseFloat(mtm[2]), qty: parseInt(mtm[3]), buyer: mtm[4], seller: mtm[5] });
            }
            const fvm = line.match(fvRe);
            if (fvm) {
              fvByProduct[fvm[1]] = { computedFV: parseFloat(fvm[2]), pos: parseInt(fvm[3]), cap: parseInt(fvm[4]) };
            }
            const obm = line.match(obRe);
            if (obm) {
              obByProduct[obm[1]] = { levels: parseInt(obm[2]), bidVol: parseInt(obm[3]), askVol: parseInt(obm[4]), spread: parseFloat(obm[5]) };
            }
            printLines.push(line);
          }

          if (printLines.length || Object.keys(ordersByProduct).length || Object.keys(fvByProduct).length) {
            logMap.set(globalTs, { printLines, ordersByProduct, ordByProduct, mtByProduct, fvByProduct, obByProduct });
          }
        }

        for (const product of Store.products) {
          for (const snap of (Store.data[product] || [])) {
            const entry = logMap.get(snap.ts);
            if (!entry) continue;
            if (entry.printLines.length) snap.logLines = entry.printLines;
            if (entry.ordersByProduct[product]) snap.botOrders = entry.ordersByProduct[product];
            snap.sentOrders = entry.ordByProduct[product] || [];
            snap.marketTrades = entry.mtByProduct[product] || [];
            if (entry.fvByProduct[product]) {
              snap.computedFV = entry.fvByProduct[product].computedFV;
              snap.remainingCapacity = entry.fvByProduct[product].cap;
            }
            if (entry.obByProduct[product]) {
              snap.obDepthSummary = entry.obByProduct[product];
            }
          }
        }
      }

      // 6. Derive per-product position and attach ownTrades from SUBMISSION fills
      if (Store.sourceType === 'official') {
        const submissionTrades = Store.trades
          .filter(t => t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION')
          .sort((a, b) => a.ts - b.ts);

        for (const product of Store.products) {
          const snaps = Store.data[product];
          if (!snaps || !snaps.length) continue;

          const prodTrades = submissionTrades.filter(t => t.symbol === product);
          let pos = 0, tradeIdx = 0;

          for (let i = 0; i < snaps.length; i++) {
            const snap = snaps[i];
            const snapOwnTrades = [];

            while (tradeIdx < prodTrades.length && prodTrades[tradeIdx].ts <= snap.ts) {
              const t = prodTrades[tradeIdx];
              if (t.buyer === 'SUBMISSION') pos += t.qty;
              else pos -= t.qty;
              snapOwnTrades.push({ price: t.price, qty: t.qty, buyer: t.buyer, seller: t.seller, ts: t.ts });
              tradeIdx++;
            }
            snap.pos = pos;
            snap.ownTrades = snapOwnTrades;
          }
        }
      }
    }

    // =====================================================================
    // SECTION 26 — FILE LOADING
    // =====================================================================
