// SECTION 25 — DATA PARSING (trades)

function parseTradesCSV(text, filename = '') {
      const inferredDay = dayFromFilename(filename);
      const dayOffset = inferredDay != null ? (Store.dayOffsets[inferredDay] || 0) : 0;

      const lines = text.trim().split('\n');
      const si = lines[0].startsWith('timestamp;') ? 1 : 0;
      for (let i = si; i < lines.length; i++) {
        const f = lines[i].split(';');
        if (f.length < 7) continue;
        const rawTs = parseInt(f[0]);
        if (isNaN(rawTs)) continue;
        const ts = rawTs + dayOffset; // apply day offset so trades align with global price timestamps
        Store.trades.push({ ts, buyer: f[1] || '', seller: f[2] || '', symbol: f[3], price: parseFloat(f[5]), qty: parseInt(f[6]) });
      }
    }

    // Trade History (lenient JSON with trailing commas)
    // Trade History timestamps are LOCAL per day (0–999900).  Apply day offsets
    // the same way parseSandboxLogs does: detect rollover when ts resets backward.

function parseTradeHistory(text) {
      const cleaned = text.replace(/,(\s*[}\]])/g, '$1');
      try {
        const arr = JSON.parse(cleaned);
        const names = new Set(Store.traderNames);
        const sortedDays = ([...Store.days]).sort((a, b) => a - b);
        let dayIdx = 0, prevLocalTs = -1;
        for (const t of arr) {
          const localTs = t.timestamp;
          // Detect day boundary: timestamp rolled back to a smaller value
          if (localTs < prevLocalTs && dayIdx < sortedDays.length - 1) dayIdx++;
          prevLocalTs = localTs;
          const day = sortedDays[dayIdx] ?? sortedDays[0];
          const globalTs = (Store.dayOffsets[day] || 0) + localTs;
          Store.trades.push({ ts: globalTs, buyer: t.buyer || '', seller: t.seller || '', symbol: t.symbol, price: t.price, qty: t.quantity });
          // Collect unique trader names (local format only — official JSON uses "" for non-SUBMISSION)
          if (t.buyer && t.buyer !== 'SUBMISSION') names.add(t.buyer);
          if (t.seller && t.seller !== 'SUBMISSION') names.add(t.seller);
        }
        Store.traderNames = [...names].sort();
        Store.sourceType = 'local';
      } catch (e) { console.warn('Trade history parse error:', e.message); }
    }

    // Sandbox logs (enrich snapshots with bot orders, positions, log lines)
