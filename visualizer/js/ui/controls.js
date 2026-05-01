// =====================================================================
    // SECTION 24 — CONTROL HANDLERS
    // =====================================================================
    document.getElementById('productSelect').addEventListener('change', e => {
      currentProduct = e.target.value;
      signalsCache = null; dynamicsCache = null; botIntelCache = null;
      autoFitViewport(); scheduleRender();
      const activeTab = document.querySelector('.tab-bar button.active');
      if (activeTab) activeTab.click();
    });
    document.getElementById('daySelect').addEventListener('change', e => {
      currentDay = e.target.value; signalsCache = null; dynamicsCache = null; botIntelCache = null; autoFitViewport(); scheduleRender();
    });
    document.getElementById('normSelect').addEventListener('change', e => {
      normMode = e.target.value; autoFitViewport(); scheduleRender();
    });
    const EMA_FV_METHODS = new Set(['emaOuterWallMid', 'emaBookVwap', 'emaMicroprice']);
    document.getElementById('fvMethodSelect').addEventListener('change', e => {
      fvMethodOverride = e.target.value;
      // Show EMA alpha only for EMA-based methods
      document.getElementById('emaAlphaParams').style.display = EMA_FV_METHODS.has(fvMethodOverride) ? '' : 'none';
      // Show Kalman params only for Kalman
      document.getElementById('kalmanParams').style.display = fvMethodOverride === 'kalmanOU' ? '' : 'none';
      // Show OU Prior params
      document.getElementById('ouPriorParams').style.display = fvMethodOverride === 'ouPrior' ? '' : 'none';
      // Show OBI lambda param
      document.getElementById('obiParams').style.display = fvMethodOverride === 'obiAdjusted' ? '' : 'none';
      recomputeAllFairValues();
      recomputeAllPnl();
      signalsCache = null; dynamicsCache = null; botIntelCache = null;
      autoFitViewport(); scheduleRender();
      const activeTab = document.querySelector('.tab-bar button.active');
      if (activeTab) activeTab.click();
    });
    document.getElementById('fvEmaAlphaInput').addEventListener('change', e => {
      const parsed = parseFloat(e.target.value);
      fvEmaAlpha = Number.isFinite(parsed) ? Math.min(0.99, Math.max(0.01, parsed)) : 0.35;
      e.target.value = fvEmaAlpha.toFixed(2);
      recomputeAllFairValues();
      recomputeAllPnl();
      signalsCache = null; dynamicsCache = null; botIntelCache = null;
      autoFitViewport(); scheduleRender();
      const activeTab = document.querySelector('.tab-bar button.active');
      if (activeTab) activeTab.click();
    });
    // Kalman parameter inputs
    ['fvKalmanKappaInput', 'fvKalmanQInput', 'fvKalmanRInput'].forEach(id => {
      document.getElementById(id)?.addEventListener('change', e => {
        const v = parseFloat(e.target.value);
        if (!Number.isFinite(v) || v <= 0) return;
        if (id === 'fvKalmanKappaInput') fvKalmanKappa = Math.min(1, v);
        else if (id === 'fvKalmanQInput') fvKalmanQ = v;
        else fvKalmanR = v;
        recomputeAllFairValues();
        recomputeAllPnl();
        signalsCache = null; dynamicsCache = null; botIntelCache = null;
        autoFitViewport(); scheduleRender();
        const activeTab = document.querySelector('.tab-bar button.active');
        if (activeTab) activeTab.click();
      });
    });
    // OU Prior parameter inputs
    document.getElementById('fvOuPriorAlphaInput')?.addEventListener('change', e => {
      const v = parseFloat(e.target.value);
      fvOuPriorAlpha = Number.isFinite(v) ? Math.min(0.99, Math.max(0.01, v)) : 0.35;
      e.target.value = fvOuPriorAlpha.toFixed(2);
      recomputeAllFairValues(); recomputeAllPnl();
      signalsCache = null; dynamicsCache = null; botIntelCache = null;
      autoFitViewport(); scheduleRender();
      const activeTab = document.querySelector('.tab-bar button.active');
      if (activeTab) activeTab.click();
    });
    document.getElementById('fvOuPriorKappaInput')?.addEventListener('change', e => {
      const v = parseFloat(e.target.value);
      if (!Number.isFinite(v) || v <= 0) return;
      fvOuPriorKappa = Math.min(1, v);
      recomputeAllFairValues(); recomputeAllPnl();
      signalsCache = null; dynamicsCache = null; botIntelCache = null;
      autoFitViewport(); scheduleRender();
      const activeTab = document.querySelector('.tab-bar button.active');
      if (activeTab) activeTab.click();
    });
    // OBI lambda input
    document.getElementById('fvObiLambdaInput')?.addEventListener('change', e => {
      const v = parseFloat(e.target.value);
      fvObiLambda = Number.isFinite(v) && v >= 0 ? v : 0.5;
      e.target.value = fvObiLambda.toFixed(2);
      recomputeAllFairValues(); recomputeAllPnl();
      signalsCache = null; dynamicsCache = null; botIntelCache = null;
      autoFitViewport(); scheduleRender();
      const activeTab = document.querySelector('.tab-bar button.active');
      if (activeTab) activeTab.click();
    });

    document.getElementById('downsample').addEventListener('input', e => {
      maxPoints = parseInt(e.target.value);
      document.getElementById('dsLabel').textContent = maxPoints >= 1000 ? (maxPoints / 1000).toFixed(0) + 'k' : maxPoints;
      scheduleRender();
    });
    document.getElementById('mmSizeInput').addEventListener('change', e => {
      mmSize = parseInt(e.target.value) || 15;
      // mmSize no longer affects FV (OuterWallMid uses outermost levels, no volume filter)
    });
    document.getElementById('exportBtn').addEventListener('click', exportAnalysis);
    document.getElementById('guideBtn').addEventListener('click', () => document.getElementById('guideModal').classList.add('show'));

    // Trade type toggle buttons
    document.querySelectorAll('.toggle-btn[data-type]').forEach(btn => {
      btn.addEventListener('click', () => {
        const type = btn.dataset.type;
        tradeTypeActive[type] = !tradeTypeActive[type];
        btn.classList.toggle('active', tradeTypeActive[type]);
        scheduleRender();
      });
    });

    // MM quote visibility toggle (stars/diamonds for make/clear orders)
    document.getElementById('tf-MQ')?.addEventListener('click', () => {
      showBotQuotes = !showBotQuotes;
      document.getElementById('tf-MQ').classList.toggle('active', showBotQuotes);
      scheduleRender();
    });

    // OB level toggles
    document.querySelectorAll('.toggle-btn[data-level]').forEach(btn => {
      btn.addEventListener('click', () => {
        const lvl = parseInt(btn.dataset.level);
        obLevelsActive[lvl] = !obLevelsActive[lvl];
        btn.classList.toggle('active', obLevelsActive[lvl]);
        scheduleRender();
      });
    });

    // Trade log filters
    ['filterSide', 'fQtyMin', 'fQtyMax', 'fDistMin', 'filterTrader'].forEach(id => {
      document.getElementById(id)?.addEventListener('change', updateTradeLog);
    });

    document.getElementById('highlightQtyMin').addEventListener('input', scheduleRender);
