// SECTION 25 — DATA PARSING (logs)

function parseSandboxLogs(text) {
      const start = text.indexOf('{');
      if (start < 0) return;
      const body = text.substring(start);

      // Extract JSON objects via brace-depth counting (string-aware)
      const spans = [];
      let depth = 0, inStr = false, esc = false, objStart = -1;
      for (let i = 0; i < body.length; i++) {
        const c = body[i];
        if (esc) { esc = false; continue; }
        if (c === '\\' && inStr) { esc = true; continue; }
        if (c === '"') { inStr = !inStr; continue; }
        if (inStr) continue;
        if (c === '{') { if (depth === 0) objStart = i; depth++; }
        if (c === '}') { depth--; if (depth === 0 && objStart >= 0) { spans.push({ s: objStart, e: i + 1 }); objStart = -1; } }
      }

      const enrichMap = new Map();
      // Detect day boundaries: each day resets localTs from ~999900 back to 0
      // We use Store.days (populated by parsePricesCSV which runs first) to assign global ts
      const sortedDays = ([...Store.days]).sort((a, b) => a - b);
      let dayIdx = 0;
      let prevLocalTs = -1;

      for (const sp of spans) {
        try {
          const obj = JSON.parse(body.substring(sp.s, sp.e));
          const ll = obj.lambdaLog || '';
          const lines = ll.split('\n');

          // Find last line that is the JSON array (old format has compressed state)
          let stateData = null, printLines = [];
          for (let j = lines.length - 1; j >= 0; j--) {
            const ln = lines[j].trim();
            if (ln.startsWith('[') && ln.endsWith(']')) {
              try { stateData = JSON.parse(ln); printLines = lines.slice(0, j).filter(l => l.trim()); break; } catch (e) { }
            }
          }

          let localTs, compOrders, position, ownTrades;
          if (stateData) {
            // OLD FORMAT: extract from compressed state
            const cs = stateData[0];
            localTs = cs[0];
            compOrders = stateData[1] || [];
            position = cs[6] || {};
            ownTrades = cs[4] || [];
          } else {
            // NEW FORMAT: no compressed state — use top-level timestamp
            localTs = obj.timestamp;
            compOrders = [];
            position = null;
            ownTrades = [];
            printLines = lines.filter(l => l.trim());
          }

          // Advance day if timestamp rolled back (new day started)
          if (localTs < prevLocalTs && dayIdx < sortedDays.length - 1) dayIdx++;
          prevLocalTs = localTs;

          const day = sortedDays[dayIdx];
          const globalTs = (Store.dayOffsets[day] || 0) + localTs;

          const botByProd = {};
          for (const o of compOrders) { if (!botByProd[o[0]]) botByProd[o[0]] = []; botByProd[o[0]].push({ price: o[1], qty: o[2] }); }
          // Overwrite botByProd with ORD| data if available (has type info for make/clear)
          // Done below after ordByProd is built

          const ownByProd = {};
          for (const t of ownTrades) { if (!ownByProd[t[0]]) ownByProd[t[0]] = []; ownByProd[t[0]].push({ price: t[1], qty: t[2], buyer: t[3], seller: t[4], ts: t[5] }); }

          // Parse FV/OB/ORD/MT structured lines from printLines
          const fvRe = /^FV\|(\w+)\|([\d.]+)\|POS\|(-?\d+)\|CAP\|(\d+)/;
          const obRe = /^OB\|(\w+)\|(\d+)LVL\|BVOL=(\d+)\|AVOL=(\d+)\|SPR=(-?[\d.]+)/;
          const ordRe = /^ORD\|(\w+)\|(TAKE|MAKE|CLEAR)\|(BUY|SELL)\|([\d.]+)\|(\d+)/;
          const mtRe = /^MT\|(\w+)\|([\d.]+)\|(\d+)\|([^|]*)\|([^|]*)/;
          const fvByProd = {}, obByProd = {}, ordByProd = {}, mtByProd = {};
          for (const line of printLines) {
            const fvm = line.match(fvRe);
            if (fvm) fvByProd[fvm[1]] = { computedFV: parseFloat(fvm[2]), pos: parseInt(fvm[3]), cap: parseInt(fvm[4]) };
            const obm = line.match(obRe);
            if (obm) obByProd[obm[1]] = { levels: parseInt(obm[2]), bidVol: parseInt(obm[3]), askVol: parseInt(obm[4]), spread: parseFloat(obm[5]) };
            const ordm = line.match(ordRe);
            if (ordm) {
              const prod = ordm[1];
              if (!ordByProd[prod]) ordByProd[prod] = [];
              ordByProd[prod].push({ type: ordm[2].toLowerCase(), side: ordm[3].toLowerCase(), price: parseFloat(ordm[4]), qty: parseInt(ordm[5]) });
            }
            const mtm = line.match(mtRe);
            if (mtm) {
              const prod = mtm[1];
              if (!mtByProd[prod]) mtByProd[prod] = [];
              mtByProd[prod].push({ price: parseFloat(mtm[2]), qty: parseInt(mtm[3]), buyer: mtm[4], seller: mtm[5] });
            }
          }

          // If ORD| lines were found, rebuild botByProd with type info for make/clear orders
          for (const prod in ordByProd) {
            const makeClears = ordByProd[prod].filter(o => o.type !== 'take');
            if (makeClears.length) {
              botByProd[prod] = makeClears.map(o => ({ price: o.price, qty: o.side === 'sell' ? -o.qty : o.qty, type: o.type }));
            }
          }

          enrichMap.set(globalTs, { position, botByProd, ownByProd, printLines, fvByProd, obByProd, ordByProd, mtByProd });
        } catch (e) { }
      }

      // Apply enrichment to snapshots
      for (const product of Store.products) {
        for (const snap of (Store.data[product] || [])) {
          const e = enrichMap.get(snap.ts);
          if (!e) continue;
          if (e.position != null) snap.pos = (e.position[product] || 0);
          snap.botOrders = e.botByProd[product] || [];
          snap.ownTrades = e.ownByProd[product] || [];
          snap.logLines = e.printLines;
          snap.sentOrders = e.ordByProd[product] || [];
          snap.marketTrades = e.mtByProd[product] || [];
          if (e.fvByProd[product]) {
            snap.computedFV = e.fvByProd[product].computedFV;
            snap.remainingCapacity = e.fvByProd[product].cap;
          }
          if (e.obByProd[product]) {
            snap.obDepthSummary = e.obByProd[product];
          }
        }
      }
    }

    // Main output.log parser (3-section splitter)

function parseOutputLog(text) {
      const actMarker = '\nActivities log:\n';
      const thMarker = '\nTrade History:\n';
      const actIdx = text.indexOf(actMarker);
      const thIdx = text.indexOf(thMarker);

      let sandboxText = '', activitiesText = '', tradeHistText = '';
      if (actIdx >= 0 && thIdx >= 0) {
        sandboxText = text.substring(0, actIdx);
        activitiesText = text.substring(actIdx + actMarker.length, thIdx);
        tradeHistText = text.substring(thIdx + thMarker.length);
      } else if (actIdx >= 0) {
        sandboxText = text.substring(0, actIdx);
        activitiesText = text.substring(actIdx + actMarker.length);
      } else {
        sandboxText = text;
      }

      if (activitiesText.trim()) parsePricesCSV(activitiesText);
      if (tradeHistText.trim()) parseTradeHistory(tradeHistText);
      if (sandboxText.trim()) parseSandboxLogs(sandboxText);

      // Reconstruct position & ownTrades from Trade History when sandbox logs
      // lack compressed state (new backtest format has no embedded state array).
      // Always run for products that have any null-pos snaps; only overwrite
      // snaps whose pos is still null so sandbox compressed-state positions are
      // kept where they exist.
      const needsPosReconstruction = Store.products.some(p =>
        (Store.data[p] || []).some(s => s.pos == null)
      );
      if (needsPosReconstruction && Store.trades.length) {
        const submissionTrades = Store.trades
          .filter(t => t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION')
          .sort((a, b) => a.ts - b.ts);

        for (const product of Store.products) {
          const snaps = Store.data[product];
          if (!snaps || !snaps.length) continue;
          // Skip products where every snap already has pos set
          if (snaps.every(s => s.pos != null)) continue;

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
            // Only overwrite pos if not already set by sandbox compressed state
            if (snap.pos == null) snap.pos = pos;
            if (!snap.ownTrades || !snap.ownTrades.length) snap.ownTrades = snapOwnTrades;
          }
        }
      }
    }

    // Official IMC Prosperity website JSON format adapter
    // Fields: activitiesLog (CSV), tradeHistory[], graphLog (CSV), logs[], positions[], profit
