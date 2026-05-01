// =====================================================================
    // SECTION 4 — FAIR VALUE & PRICE TRANSFORM
    // =====================================================================
    // Fair value model framework.
    // To add a new FV model:
    // 1) Add a constant in FAIR_VALUE_METHODS
    // 2) Add a case in computeFairValueForSnapshot(...)
    // 3) Add an option in #fvMethodSelect (controls bar)
    function resolveFairValueMethod(product) {
      if (fvMethodOverride !== FAIR_VALUE_METHODS.PRODUCT_DEFAULT) return fvMethodOverride;
      const cfg = PRODUCT_CONFIG[product];
      return cfg?.fairValueModel || FAIR_VALUE_METHODS.OUTER_WALL_MID;
    }

    function computeOuterWallMid(bids, asks) {
      if (!(bids?.length && asks?.length)) return null;
      const bidWall = Math.min(...bids.map(b => b.price));
      const askWall = Math.max(...asks.map(a => a.price));
      return (bidWall + askWall) / 2;
    }

    function computeRawMid(bids, asks) {
      if (!(bids?.length && asks?.length)) return null;
      const bestBid = Math.max(...bids.map(b => b.price));
      const bestAsk = Math.min(...asks.map(a => a.price));
      return (bestBid + bestAsk) / 2;
    }

    function computeBookVwap(bids, asks) {
      // Depth-weighted average: bid_vwap and ask_vwap computed separately,
      // then averaged — averaging both sides cancels imbalance bias.
      // Returns null on one-sided/empty book (gated).
      if (!(bids?.length && asks?.length)) return null;
      const bidVol = bids.reduce((s, b) => s + b.volume, 0);
      const askVol = asks.reduce((s, a) => s + a.volume, 0);
      if (bidVol === 0 || askVol === 0) return null;
      const bidVwap = bids.reduce((s, b) => s + b.price * b.volume, 0) / bidVol;
      const askVwap = asks.reduce((s, a) => s + a.price * a.volume, 0) / askVol;
      return (bidVwap + askVwap) / 2;
    }

    function computeMicroprice(bids, asks) {
      // Stoikov microprice: L1 imbalance-weighted mid.
      // microprice = best_bid + spread × (ask_vol_L1 / (bid_vol_L1 + ask_vol_L1))
      // When ask side is heavier → price pulled toward ask (momentum signal).
      // Returns null on one-sided/empty book.
      if (!(bids?.length && asks?.length)) return null;
      const bestBid = Math.max(...bids.map(b => b.price));
      const bestAsk = Math.min(...asks.map(a => a.price));
      const bidVol = bids.find(b => b.price === bestBid)?.volume ?? 0;
      const askVol = asks.find(a => a.price === bestAsk)?.volume ?? 0;
      const totalVol = bidVol + askVol;
      if (totalVol === 0) return (bestBid + bestAsk) / 2;
      return bestBid + (bestAsk - bestBid) * (askVol / totalVol);
    }

    function computeFairValueForSnapshot(product, snap, state) {
      const cfg = PRODUCT_CONFIG[product] || {};
      if (cfg.fairValue != null) {
        state.prevFair = cfg.fairValue;
        return cfg.fairValue;
      }

      const method = resolveFairValueMethod(product);
      const anchor = cfg.anchorFairValue ?? 10000;
      const outerWallMid = computeOuterWallMid(snap.bids, snap.asks);
      const rawMid = computeRawMid(snap.bids, snap.asks);

      let fair = null;
      switch (method) {
        case FAIR_VALUE_METHODS.OUTER_WALL_MID:
          fair = outerWallMid ?? state.prevFair ?? anchor;
          break;
        case FAIR_VALUE_METHODS.EMA_OUTER_WALL_MID:
          if (outerWallMid != null) {
            state.ema = state.ema == null ? outerWallMid : fvEmaAlpha * outerWallMid + (1 - fvEmaAlpha) * state.ema;
          }
          fair = state.ema ?? state.prevFair ?? anchor;
          break;
        case FAIR_VALUE_METHODS.KALMAN_OU: {
          // Kalman filter with Ornstein-Uhlenbeck mean-reversion prior
          const kx = state.kalman_x ?? anchor;
          const kP = state.kalman_P ?? fvKalmanR;
          const kappa = fvKalmanKappa;
          // Predict: OU drift toward anchor
          const x_pred = (1 - kappa) * kx + kappa * anchor;
          const P_pred = (1 - kappa) * (1 - kappa) * kP + fvKalmanQ;
          // Update: only on two-sided ticks
          if (outerWallMid != null) {
            const K = P_pred / (P_pred + fvKalmanR);
            state.kalman_x = x_pred + K * (outerWallMid - x_pred);
            state.kalman_P = (1 - K) * P_pred;
          } else {
            state.kalman_x = x_pred;
            state.kalman_P = P_pred;
          }
          fair = state.kalman_x;
          break;
        }
        case FAIR_VALUE_METHODS.OU_PRIOR: {
          // OU Prior: fuses OU mean-reversion prior with current OuterWallMid observation.
          // Prior: drift = (1-κ)*prev + κ*anchor  (OU pull toward anchor)
          // Update: state = α*observation + (1-α)*drift  (trust market when available)
          const ouPriorPrev = state.ou_prior ?? anchor;
          const drift = (1 - fvOuPriorKappa) * ouPriorPrev + fvOuPriorKappa * anchor;
          if (outerWallMid != null) {
            state.ou_prior = fvOuPriorAlpha * outerWallMid + (1 - fvOuPriorAlpha) * drift;
          } else {
            state.ou_prior = drift;
          }
          fair = state.ou_prior;
          break;
        }
        case FAIR_VALUE_METHODS.OBI_ADJUSTED: {
          // OBI-Adjusted: OuterWallMid + λ * halfSpread * OBI
          // OBI = (bid_vol_L1 - ask_vol_L1) / (bid_vol_L1 + ask_vol_L1)  ∈ [-1, +1]
          // Positive OBI (bid-heavy) shifts FV up; negative (ask-heavy) shifts down.
          const base = outerWallMid ?? state.prevFair ?? anchor;
          if (snap.bids?.length && snap.asks?.length) {
            const bestBid = Math.max(...snap.bids.map(b => b.price));
            const bestAsk = Math.min(...snap.asks.map(a => a.price));
            const bidVol = snap.bids.find(b => b.price === bestBid)?.vol ?? 0;
            const askVol = snap.asks.find(a => a.price === bestAsk)?.vol ?? 0;
            const totalVol = bidVol + askVol;
            const obi = totalVol > 0 ? (bidVol - askVol) / totalVol : 0;
            const halfSpread = Math.max(0, (bestAsk - bestBid) / 2);
            fair = base + fvObiLambda * halfSpread * obi;
          } else {
            fair = base;
          }
          break;
        }
        case FAIR_VALUE_METHODS.RAW_MID:
          fair = rawMid ?? state.prevFair ?? anchor;
          break;
        case FAIR_VALUE_METHODS.BOOK_VWAP: {
          // Raw book VWAP — no smoothing, gated on two-sided ticks
          const bvwap = computeBookVwap(snap.bids, snap.asks);
          fair = bvwap ?? state.prevFair ?? anchor;
          break;
        }
        case FAIR_VALUE_METHODS.EMA_BOOK_VWAP: {
          // EMA-smoothed book VWAP — matches trader4.py logic exactly
          const bvwap = computeBookVwap(snap.bids, snap.asks);
          if (bvwap != null) {
            state.vwap_ema = state.vwap_ema == null ? bvwap : fvEmaAlpha * bvwap + (1 - fvEmaAlpha) * state.vwap_ema;
          }
          fair = state.vwap_ema ?? state.prevFair ?? anchor;
          break;
        }
        case FAIR_VALUE_METHODS.MICROPRICE: {
          // Raw Stoikov microprice — L1 imbalance-weighted mid, no smoothing
          const mp = computeMicroprice(snap.bids, snap.asks);
          fair = mp ?? state.prevFair ?? anchor;
          break;
        }
        case FAIR_VALUE_METHODS.EMA_MICROPRICE: {
          // EMA-smoothed microprice — dampens the fast L1 oscillation
          const mp = computeMicroprice(snap.bids, snap.asks);
          if (mp != null) {
            state.micro_ema = state.micro_ema == null ? mp : fvEmaAlpha * mp + (1 - fvEmaAlpha) * state.micro_ema;
          }
          fair = state.micro_ema ?? state.prevFair ?? anchor;
          break;
        }
        default:
          fair = outerWallMid ?? state.prevFair ?? anchor;
          break;
      }

      state.prevFair = fair;
      return fair;
    }

    function recomputeFairValuesForProduct(product) {
      const snaps = Store.data[product] || [];
      if (!snaps.length) return;

      let state = { prevFair: null, ema: null, kalman_x: null, kalman_P: null, vwap_ema: null, micro_ema: null, ou_prior: null, day: null };
      for (const snap of snaps) {
        // Reset smoothing state on day changes to avoid leaking across day boundaries.
        if (state.day !== snap.day) state = { prevFair: null, ema: null, kalman_x: null, kalman_P: null, vwap_ema: null, micro_ema: null, ou_prior: null, day: snap.day };
        snap.fair = computeFairValueForSnapshot(product, snap, state);
      }
    }

    function recomputeAllFairValues() {
      for (const product of Store.products) recomputeFairValuesForProduct(product);
    }

    function getFairValueMethodNote(product) {
      const cfg = PRODUCT_CONFIG[product];
      if (cfg?.fairValue != null) return `Fixed fair value: ${cfg.fairValue}`;
      const method = resolveFairValueMethod(product);
      if (method === FAIR_VALUE_METHODS.EMA_OUTER_WALL_MID) {
        return `EMA(OuterWallMid): EMA_t = ${fvEmaAlpha.toFixed(2)}*OuterWallMid_t + ${(1 - fvEmaAlpha).toFixed(2)}*EMA_(t-1); OuterWallMid=(min_bid+max_ask)/2`;
      }
      if (method === FAIR_VALUE_METHODS.KALMAN_OU) {
        return `Kalman(OU): κ=${fvKalmanKappa}, Q=${fvKalmanQ}, R=${fvKalmanR}, anchor=${(PRODUCT_CONFIG[product]?.anchorFairValue ?? 10000)}. Updates only on two-sided ticks; drifts toward anchor on one-sided.`;
      }
      if (method === FAIR_VALUE_METHODS.OU_PRIOR) {
        return `OU Prior: drift = (1-κ)*prev + κ*anchor; update = α*OuterWallMid + (1-α)*drift. α=${fvOuPriorAlpha.toFixed(2)}, κ=${fvOuPriorKappa.toFixed(3)}, anchor=${(PRODUCT_CONFIG[product]?.anchorFairValue ?? 10000)}. Resets per day.`;
      }
      if (method === FAIR_VALUE_METHODS.OBI_ADJUSTED) {
        return `OBI-Adjusted: OuterWallMid + λ × halfSpread × OBI. OBI = (bidVol_L1 − askVol_L1)/(bidVol_L1 + askVol_L1). λ=${fvObiLambda.toFixed(2)}. Positive OBI (bid-heavy) shifts FV up.`;
      }
      if (method === FAIR_VALUE_METHODS.RAW_MID) return 'RawMid: (best_bid + best_ask) / 2';
      if (method === FAIR_VALUE_METHODS.BOOK_VWAP) return 'BookVwap: (bid_vwap + ask_vwap)/2 — each side volume-weighted across all levels. Gated on two-sided ticks (holds on one-sided).';
      if (method === FAIR_VALUE_METHODS.EMA_BOOK_VWAP) return `EMA(BookVwap): α=${fvEmaAlpha.toFixed(2)} — smoothed book VWAP. Matches trader4.py. Gated: no update on one-sided/empty ticks.`;
      if (method === FAIR_VALUE_METHODS.MICROPRICE) return 'Microprice (Stoikov): best_bid + spread × (ask_vol_L1 / (bid_vol_L1 + ask_vol_L1)). L1 imbalance-weighted mid. Gated on two-sided.';
      if (method === FAIR_VALUE_METHODS.EMA_MICROPRICE) return `EMA(Microprice): α=${fvEmaAlpha.toFixed(2)} — smoothed L1 microprice. Best for oscillating books (Osmium).`;
      return 'OuterWallMid: (min_bid + max_ask) / 2';
    }

    // =====================================================================
    // MTM PnL reconstruction from SUBMISSION trades + current FV
    // =====================================================================
    // Computes mark-to-market PnL per snapshot: cash_flow + position * FV.
    // Stores result in snap.mtmPnl.  Called after trades are loaded and after
    // FV changes so the PnL chart always reflects the current FV model.
    function recomputePnlForProduct(product) {
      const snaps = Store.data[product];
      if (!snaps || !snaps.length) return;

      const submissionTrades = (Store.trades || [])
        .filter(t => t.symbol === product && (t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION'))
        .sort((a, b) => a.ts - b.ts);

      if (!submissionTrades.length) return;  // no own trades → nothing to mark

      let cash = 0, pos = 0, tradeIdx = 0;
      for (const snap of snaps) {
        while (tradeIdx < submissionTrades.length && submissionTrades[tradeIdx].ts <= snap.ts) {
          const t = submissionTrades[tradeIdx++];
          if (t.buyer === 'SUBMISSION') { cash -= t.price * t.qty; pos += t.qty; }
          else                          { cash += t.price * t.qty; pos -= t.qty; }
        }
        snap.mtmPnl = snap.fair != null ? cash + pos * snap.fair : null;
      }
    }

    function recomputeAllPnl() {
      for (const product of Store.products) recomputePnlForProduct(product);
    }

    function transformPrice(price, fair) {
      if (fair == null || price == null) return price;
      if (normMode === 'fair') return price - fair;
      if (normMode === 'pct') return ((price - fair) / fair) * 100;
      return price;
    }
