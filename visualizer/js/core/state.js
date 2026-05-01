// =====================================================================
    // SECTION 2 — MUTABLE STATE
    // =====================================================================
    let mmSize = 15;       // Legacy: no longer used for FV (OuterWallMid uses outermost levels)
    let fvMethodOverride = FAIR_VALUE_METHODS.PRODUCT_DEFAULT;
    let fvEmaAlpha = 0.35;
    let fvKalmanKappa = 0.05;  // OU mean-reversion speed (higher = faster pull toward anchor)
    let fvKalmanQ = 1.0;       // Process noise (how much true value moves per tick)
    let fvKalmanR = 50.0;      // Observation noise (wall_mid jitter variance)
    let fvOuPriorAlpha = 0.35; // OU Prior: weight on current OuterWallMid observation
    let fvOuPriorKappa = 0.05; // OU Prior: mean-reversion speed toward anchor
    let fvObiLambda = 0.5;     // OBI-Adjusted: scale factor for imbalance shift
    let normMode = 'raw';
    let maxPoints = 3000;
    let hoverTs = null;    // timestamp under mouse cursor (for crosshair)
    let signalsCache = null; // Cached spike/trader analysis, invalidated on product/day change
    let dynamicsCache = null; // Cached dynamics analysis (OU, regimes, OFI), invalidated on product/day change
    let botIntelCache = null; // Cached bot intelligence analysis, invalidated on product/day change

    // Trade type display toggles
    const tradeTypeActive = { TB: true, TS: true, MB: true, MS: true, CB: true, CS: true, M: true, S: true, B: true };
    // MM quote display toggle (bot orders: make/clear quote stars and diamonds)
    let showBotQuotes = true;
    // OB level display toggles (index 0=L1, 1=L2, 2=L3)
    const obLevelsActive = [true, true, true];

    // Data store
    const Store = {
      data: {},           // product -> sorted array of snapshots
      trades: [],         // all trades (from trade CSV / trade history)
      products: [],
      days: [],
      dayOffsets: {},
      graphLog: [],       // [{ts, value}] total PnL sampled every 400 ticks (official JSON only)
      finalPositions: {}, // {product: qty} final positions (official JSON only)
      totalProfit: null,  // total profit scalar (official JSON only)
      traderNames: [],    // sorted unique trader names (local output.log format only)
      sourceType: null,   // 'local' | 'official'
      dailyRanges: {},    // {product: {day: {min, max}}} pre-computed daily price ranges
    };

    let currentProduct = '';
    let currentDay = 'all';

    // Viewport (shared X axis for both canvases)
    const vp = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
    let mainW = 1, mainH = 1, ppW = 1, ppH = 1;

    // Canvas contexts
    let mainCtx = null, ppCtx = null;

    // Interaction state
    let dragState = null;
    let tooltipTimer = null;
    let renderFrame = null;
