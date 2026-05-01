// =====================================================================
    // SECTION 3 — COORDINATE TRANSFORMS
    // =====================================================================
    const tsToX = ts => ((ts - vp.xMin) / (vp.xMax - vp.xMin)) * mainW;
    const priceToY = p => mainH - ((p - vp.yMin) / (vp.yMax - vp.yMin)) * mainH;
    const xToTs = x => vp.xMin + (x / mainW) * (vp.xMax - vp.xMin);
    const yToPrice = y => vp.yMin + ((mainH - y) / mainH) * (vp.yMax - vp.yMin);
    const ppTsToX = ts => ((ts - vp.xMin) / (vp.xMax - vp.xMin)) * ppW;

    // =====================================================================
    // SECTION 8 — CANVAS SETUP
    // =====================================================================
    function setupCanvas(canvas, container) {
      const dpr = window.devicePixelRatio || 1;
      const r = container.getBoundingClientRect();
      canvas.width = r.width * dpr;
      canvas.height = r.height * dpr;
      canvas.style.width = r.width + 'px';
      canvas.style.height = r.height + 'px';
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      return { ctx, w: r.width, h: r.height };
    }
    function initCanvases() {
      const m = setupCanvas(document.getElementById('mainCanvas'), document.getElementById('mainChartDiv'));
      mainCtx = m.ctx; mainW = m.w; mainH = m.h;
      const p = setupCanvas(document.getElementById('posPnlCanvas'), document.getElementById('posPnlDiv'));
      ppCtx = p.ctx; ppW = p.w; ppH = p.h;
    }
