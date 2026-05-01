// =====================================================================
    // SECTION 10 — VIEWPORT AUTO-FIT
    // =====================================================================
    // Full reset: fit X to all data, then Y to that X range
    function autoFitViewport() {
      const snaps = getVisSnaps();
      if (!snaps.length) return;
      vp.xMin = snaps[0].ts - 200; vp.xMax = snaps[snaps.length - 1].ts + 200;
      autoFitY();
    }

    // Fit Y to the currently visible X range (called on every render so Y always tracks)
    function autoFitY() {
      const snaps = getRange(); // only ticks inside current xMin..xMax
      if (!snaps.length) return;
      let yMin = Infinity, yMax = -Infinity;
      for (const s of snaps) {
        for (const b of s.bids) { const p = transformPrice(b.price, s.fair); if (p < yMin) yMin = p; if (p > yMax) yMax = p; }
        for (const a of s.asks) { const p = transformPrice(a.price, s.fair); if (p < yMin) yMin = p; if (p > yMax) yMax = p; }
      }
      if (yMin === Infinity) return;
      const pad = (yMax - yMin) * 0.1 || 2;
      vp.yMin = yMin - pad; vp.yMax = yMax + pad;
    }

    // =====================================================================
    // SECTION 11 — RENDER SCHEDULE
    // =====================================================================
    function scheduleRender() {
      if (renderFrame) cancelAnimationFrame(renderFrame);
      renderFrame = requestAnimationFrame(() => {
        renderMain();
        renderPosPnl();
        updateTradeLog();
        updateQuickStats();
      });
    }

    // =====================================================================
    // SECTION 12 — MAIN CHART RENDER
    // =====================================================================
    function renderMain() {
      if (!mainCtx) return;
      const ctx = mainCtx, w = mainW, h = mainH;
      ctx.clearRect(0, 0, w, h);

      if (!currentProduct || !Store.data[currentProduct]) {
        ctx.fillStyle = '#2a2a4a'; ctx.font = '12px Courier New'; ctx.textAlign = 'center';
        ctx.fillText('Drop  output.log  OR  prices/trades CSVs  to begin', w / 2, h / 2 - 16);
        ctx.fillStyle = '#444'; ctx.font = '10px Courier New';
        ctx.fillText('Click ? in the toolbar for a full usage guide', w / 2, h / 2 + 4);
        ctx.textAlign = 'start'; return;
      }

      autoFitY(); // Y always tracks the visible X range

      const snaps = getRange();
      const ds = downsample(snaps, maxPoints);
      const allSnaps = getVisSnaps();

      drawGrid(ctx, w, h, vp.xMin, vp.xMax, vp.yMin, vp.yMax);

      // Day boundary markers
      if (currentDay === 'all' && Store.days.length > 1) {
        ctx.setLineDash([4, 6]); ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
        for (const d of Store.days) {
          const off = Store.dayOffsets[d];
          if (off > 0) {
            const x = tsToX(off);
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
            ctx.fillStyle = '#444'; ctx.font = '9px Courier New'; ctx.fillText('Day ' + d, x + 3, 13);
          }
        }
        ctx.setLineDash([]);
      }

      // Fair value line
      if (normMode === 'raw') {
        const cfg = PRODUCT_CONFIG[currentProduct];
        if (cfg && cfg.fairValue !== null) {
          const y = priceToY(cfg.fairValue);
          ctx.strokeStyle = COLORS.fairLine; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = COLORS.fairLine; ctx.font = '9px Courier New';
          ctx.fillText('FV=' + cfg.fairValue, 4, y - 3);
        } else if (ds.length > 1) {
          ctx.strokeStyle = COLORS.fairLine; ctx.lineWidth = 1; ctx.globalAlpha = 0.4; ctx.setLineDash([4, 4]);
          ctx.beginPath(); let st = false;
          for (const s of ds) { if (s.fair == null) continue; const x = tsToX(s.ts), y = priceToY(s.fair); if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y); }
          ctx.stroke(); ctx.globalAlpha = 1; ctx.setLineDash([]);
        }
      } else {
        // In normalized modes, fair value is at 0
        const y = priceToY(0);
        ctx.strokeStyle = COLORS.fairLine; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        ctx.setLineDash([]);
      }

      // Bid/ask continuous lines per OB level (L1 solid, L2 faded, L3 dashed)
      ctx.lineJoin = 'round';
      for (let lvl = 0; lvl < 3; lvl++) {
        if (!obLevelsActive[lvl]) continue;
        const alpha = lvl === 0 ? 0.9 : lvl === 1 ? 0.55 : 0.5;
        const lw = lvl === 0 ? 1.6 : lvl === 1 ? 1.0 : 1.0;
        if (lvl === 2) ctx.setLineDash([4, 3]);

        // Bid line (blue) — L3 carries forward last known price through gaps
        ctx.strokeStyle = COLORS.bid; ctx.globalAlpha = alpha; ctx.lineWidth = lw;
        ctx.beginPath(); let bs = false; let lastBidY = null;
        for (const s of ds) {
          const x = tsToX(s.ts);
          if (s.bids[lvl]) {
            const y = priceToY(transformPrice(s.bids[lvl].price, s.fair));
            lastBidY = y;
            if (!bs) { ctx.moveTo(x, y); bs = true; } else ctx.lineTo(x, y);
          } else if (lvl === 2 && lastBidY !== null && bs) {
            ctx.lineTo(x, lastBidY);
          } else {
            bs = false;
          }
        }
        ctx.stroke();

        // Ask line (red) — L3 carries forward last known price through gaps
        ctx.strokeStyle = COLORS.ask;
        ctx.beginPath(); let as2 = false; let lastAskY = null;
        for (const s of ds) {
          const x = tsToX(s.ts);
          if (s.asks[lvl]) {
            const y = priceToY(transformPrice(s.asks[lvl].price, s.fair));
            lastAskY = y;
            if (!as2) { ctx.moveTo(x, y); as2 = true; } else ctx.lineTo(x, y);
          } else if (lvl === 2 && lastAskY !== null && as2) {
            ctx.lineTo(x, lastAskY);
          } else {
            as2 = false;
          }
        }
        ctx.stroke();
        if (lvl === 2) ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      // L1 volume ticks — small vertical nubs proportional to volume on L1 bid/ask
      if (obLevelsActive[0] && ds.length < 2000) {
        for (const s of ds) {
          if (s.bids[0]) { const x = tsToX(s.ts), y = priceToY(transformPrice(s.bids[0].price, s.fair)), h2 = Math.min(6, s.bids[0].vol / 8); ctx.fillStyle = COLORS.bid; ctx.globalAlpha = 0.3; ctx.fillRect(x - 0.5, y, 1, h2); }
          if (s.asks[0]) { const x = tsToX(s.ts), y = priceToY(transformPrice(s.asks[0].price, s.fair)), h2 = Math.min(6, s.asks[0].vol / 8); ctx.fillStyle = COLORS.ask; ctx.globalAlpha = 0.3; ctx.fillRect(x - 0.5, y - h2, 1, h2); }
        }
        ctx.globalAlpha = 1;
      }

      // Mid price line
      if (ds.length > 1) {
        ctx.strokeStyle = COLORS.midLine; ctx.lineWidth = 0.8;
        ctx.beginPath(); let st = false;
        for (const s of ds) {
          if (s.mid == null) continue;
          const x = tsToX(s.ts), y = priceToY(transformPrice(s.mid, s.fair));
          if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Bot orders (our quotes: stars for make, diamonds for clear) — colored by side
      if (showBotQuotes && ds.length < 1200) {
        for (const s of ds) {
          const x = tsToX(s.ts);
          for (const o of s.botOrders) {
            const y = priceToY(transformPrice(o.price, s.fair));
            const isBuy = o.qty > 0;
            if (o.type === 'clear') {
              drawDiamond(ctx, x, y, 5, isBuy ? COLORS.clearBidQuote : COLORS.clearAskQuote, '#0a1020');
            } else {
              drawStar(ctx, x, y, 5, isBuy ? COLORS.makeBidQuote : COLORS.makeAskQuote, '#0a1020');
            }
          }
        }
      }

      // Trade markers (classified by type)
      const visTrades = getVisTrades();
      const highlightQty = parseInt(document.getElementById('highlightQtyMin')?.value) || 0;
      const highlightTrader = document.getElementById('filterTrader')?.value || '';
      for (const t of visTrades) {
        const snap = findNearest(allSnaps, t.ts);
        const type = classifyTrade(t, snap);
        if (!tradeTypeActive[type]) continue;

        const x = tsToX(t.ts);
        const price = snap ? transformPrice(t.price, snap.fair) : t.price;
        const y = priceToY(price);
        const isLarge = highlightQty > 0 && t.qty >= highlightQty;
        const isTrader = highlightTrader && (t.buyer === highlightTrader || t.seller === highlightTrader);
        const highlighted = isLarge || isTrader;
        let color = TRADE_COLORS[type] || '#888';
        if (isTrader) color = '#ffee00';

        const ownFill = isOwnFillType(type);
        const baseR = ownFill ? Math.max(4, Math.min(8, t.qty * 0.8))
                    : type === 'S' ? Math.max(4, Math.min(8, t.qty * 1.2))
                    : type === 'B' ? Math.max(7, Math.min(13, t.qty * 1.5))
                    : Math.max(2.5, Math.min(7, t.qty / 2));
        const r = highlighted ? baseR * 2 : baseR;

        if (highlighted) {
          ctx.save();
          ctx.shadowColor = isTrader ? '#ffee00' : color;
          ctx.shadowBlur = 8;
        }

        if (type === 'TB' || type === 'TS') {
          drawCross(ctx, x, y, highlighted ? 10 : 6, color);
        } else if (type === 'MB' || type === 'MS') {
          // Fixed size: always r=8, bigger than quote stars (r=5), same for all fill sizes
          const starFill = type === 'MB' ? '#2a0060' : '#602000';
          drawStar(ctx, x, y, highlighted ? 16 : 8, color, starFill);
        } else if (type === 'CB' || type === 'CS') {
          drawFilledDiamond(ctx, x, y, r, color);
        } else if (type === 'M') {
          drawSquare(ctx, x, y, r, color);
        } else {
          // S, B — bot trade triangles
          const up = tradeBuyDir(t, snap);
          drawTriangle(ctx, x, y, r, up, color);
        }
        if (highlighted) ctx.restore();
      }

      // Selection rect (zoom drag)
      if (dragState && dragState.type === 'zoom') {
        const x1 = Math.min(dragState.sx, dragState.cx), x2 = Math.max(dragState.sx, dragState.cx);
        ctx.fillStyle = COLORS.selBox; ctx.fillRect(x1, 0, x2 - x1, h);
        ctx.strokeStyle = COLORS.selBorder; ctx.lineWidth = 1; ctx.strokeRect(x1, 0, x2 - x1, h);
      }

      // Vertical crosshair
      if (hoverTs !== null) {
        const cx2 = tsToX(hoverTs);
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.moveTo(cx2, 0); ctx.lineTo(cx2, h); ctx.stroke();
        ctx.setLineDash([]);
        // Timestamp label at top
        ctx.fillStyle = 'rgba(76,201,240,0.9)'; ctx.font = '9px Courier New'; ctx.textAlign = 'center';
        const tsLabel = 'T=' + Math.round(hoverTs);
        const lw3 = ctx.measureText(tsLabel).width + 6;
        ctx.fillRect(cx2 - lw3 / 2, 0, lw3, 14);
        ctx.fillStyle = '#000'; ctx.fillText(tsLabel, cx2, 10);
        ctx.restore();
      }

      // Corner legend
      ctx.globalAlpha = 0.82;
      const lx = w - 130, ly = 10, lh = 14, lw2 = 100;
      ctx.fillStyle = 'rgba(10,15,30,0.7)';
      ctx.fillRect(lx - 4, ly - 2, lw2 + 8, lh * 5 + 4);
      ctx.globalAlpha = 1;
      ctx.font = '8px Courier New';
      const leg = [
        [COLORS.bid, '━━ Bid L1/L2, -- L3'],
        [COLORS.ask, '━━ Ask L1/L2, -- L3'],
        [COLORS.fairLine, '-- Fair Value (selected model)'],
        ['#888', '━━ Mid price'],
        ['#fff', '★ MakeQ  ◆ ClearQ  × Take  ★ Make  ◆ Clear'],
      ];
      leg.forEach(([c, t], i) => { ctx.fillStyle = c; ctx.fillText(t, lx, ly + i * lh + 9); });
    }
