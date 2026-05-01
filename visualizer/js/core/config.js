'use strict';

    // =====================================================================
    // SECTION 1 — CONSTANTS & CONFIG
    // =====================================================================
    const FAIR_VALUE_METHODS = {
      PRODUCT_DEFAULT: 'productDefault',
      OUTER_WALL_MID: 'outerWallMid',
      EMA_OUTER_WALL_MID: 'emaOuterWallMid',
      KALMAN_OU: 'kalmanOU',
      OU_PRIOR: 'ouPrior',
      OBI_ADJUSTED: 'obiAdjusted',
      RAW_MID: 'rawMid',
      BOOK_VWAP: 'bookVwap',
      EMA_BOOK_VWAP: 'emaBookVwap',
      MICROPRICE: 'microprice',
      EMA_MICROPRICE: 'emaMicroprice',
    };
    const PRODUCT_CONFIG = {
      EMERALDS: { fairValue: 10000, posLimit: 80, color: '#4cc9f0', fairValueModel: FAIR_VALUE_METHODS.PRODUCT_DEFAULT },
      TOMATOES: { fairValue: null, posLimit: 80, color: '#ff8800', fairValueModel: FAIR_VALUE_METHODS.OUTER_WALL_MID },
      ASH_COATED_OSMIUM: { fairValue: null, posLimit: 80, color: '#44ff88', fairValueModel: FAIR_VALUE_METHODS.EMA_OUTER_WALL_MID, anchorFairValue: 10001 },
      INTARIAN_PEPPER_ROOT: { fairValue: null, posLimit: 80, color: '#ffcc44', fairValueModel: FAIR_VALUE_METHODS.OUTER_WALL_MID, anchorFairValue: 12000 },
    };
    const COLORS = {
      bid: '#4488ff', ask: '#ff4444',
      midLine: 'rgba(255,255,255,0.25)', fairLine: '#44ff88',
      botOrderBid: '#4488ff', botOrderAsk: '#ff4444',
      position: '#ffcc00', pnl: '#44ff44',
      grid: '#1e2a3a', gridText: '#3a4a5a',
      selBox: 'rgba(76,201,240,0.15)', selBorder: '#4cc9f0',
      posLimit: 'rgba(255,0,0,0.10)',
      makeBidQuote: '#ffffff', makeAskQuote: '#ffffff',  // quote stars = white both sides
      clearBidQuote: '#44ff88', clearAskQuote: '#44ff88',
    };
    const TRADE_COLORS = {
      TB: '#ffee00', TS: '#aa44ff',   // Take: yellow ask-take ×, purple bid-take ×
      MB: '#aa44ff', MS: '#ff8800',   // Make: purple buy ★, orange sell ★
      CB: '#44ff88', CS: '#44ff88',   // Clear: green both ◆
      M: '#aa88ff', S: '#88bbff', B: '#ff4488',
    };
    const SMALL_QTY_THRESHOLD = 5;  // S vs B split
