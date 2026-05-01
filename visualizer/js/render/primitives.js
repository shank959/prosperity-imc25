// =====================================================================
    // SECTION 9 — DRAWING PRIMITIVES
    // =====================================================================
    function drawDot(ctx, x, y, r, color, alpha = 0.65) {
      ctx.globalAlpha = alpha; ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }
    function drawStar(ctx, x, y, r, strokeColor = '#ffffff', fillColor = '#000000') {
      ctx.strokeStyle = strokeColor; ctx.fillStyle = fillColor; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < 5; i++) {
        const oa = (i * 72 - 90) * Math.PI / 180, ia = ((i * 72) + 36 - 90) * Math.PI / 180;
        ctx.lineTo(x + r * Math.cos(oa), y + r * Math.sin(oa));
        ctx.lineTo(x + r * .38 * Math.cos(ia), y + r * .38 * Math.sin(ia));
      }
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    function drawDiamond(ctx, x, y, r, strokeColor = '#4cc9f0', fillColor = '#0a1020') {
      ctx.strokeStyle = strokeColor; ctx.fillStyle = fillColor; ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    function drawCross(ctx, x, y, r, color) {
      ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
      ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r);
      ctx.stroke();
    }
    function drawCircle(ctx, x, y, r, color, alpha = 0.8) {
      ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    function drawFilledDiamond(ctx, x, y, r, color, alpha = 0.8) {
      ctx.globalAlpha = alpha; ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 0.8; ctx.stroke();
      ctx.globalAlpha = 1;
    }
    function drawTriangle(ctx, x, y, r, up, color, alpha = 0.7) {
      ctx.globalAlpha = alpha; ctx.fillStyle = color;
      ctx.beginPath();
      if (up) { ctx.moveTo(x, y - r); ctx.lineTo(x - r, y + r); ctx.lineTo(x + r, y + r); }
      else { ctx.moveTo(x, y + r); ctx.lineTo(x - r, y - r); ctx.lineTo(x + r, y - r); }
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 0.8; ctx.stroke();
      ctx.globalAlpha = 1;
    }
    function drawSquare(ctx, x, y, r, color, alpha = 0.75) {
      ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.strokeRect(x - r, y - r, r * 2, r * 2);
      ctx.globalAlpha = 1;
    }
    function niceInterval(range, ticks) {
      const rough = range / ticks, mag = Math.pow(10, Math.floor(Math.log10(rough || 1))), n = rough / mag;
      return (n <= 1.5 ? 1 : n <= 3.5 ? 2 : n <= 7.5 ? 5 : 10) * mag;
    }
    function drawGrid(ctx, w, h, xMin, xMax, yMin, yMax) {
      ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 0.5;
      ctx.fillStyle = COLORS.gridText; ctx.font = '9px Courier New';

      const yi = niceInterval(yMax - yMin, 7);
      for (let v = Math.ceil(yMin / yi) * yi; v <= yMax; v += yi) {
        const y = h - ((v - yMin) / (yMax - yMin)) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        let lbl = normMode === 'pct' ? v.toFixed(2) + '%' : normMode === 'fair' ? (v >= 0 ? '+' : '') + v.toFixed(1) : v.toFixed(0);
        ctx.fillText(lbl, 3, y - 2);
      }
      const xi = niceInterval(xMax - xMin, 9);
      ctx.textAlign = 'center';
      for (let v = Math.ceil(xMin / xi) * xi; v <= xMax; v += xi) {
        const x = ((v - xMin) / (xMax - xMin)) * w;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        ctx.fillText(v.toFixed(0), x, h - 3);
      }
      ctx.textAlign = 'start';
    }
