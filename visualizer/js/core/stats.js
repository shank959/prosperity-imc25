// =====================================================================
    // SECTION 6 — STATISTICS UTILS
    // =====================================================================
    const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
    const sum2 = (a, m) => a.reduce((s, v) => s + (v - m) ** 2, 0);
    const vari = a => a.length > 1 ? sum2(a, mean(a)) / (a.length - 1) : 0;
    const std = a => Math.sqrt(vari(a));
    const med = a => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
    const skew = a => { const m = mean(a), s = std(a), n = a.length; return s && n > 2 ? (n / ((n - 1) * (n - 2))) * a.reduce((acc, v) => acc + ((v - m) / s) ** 3, 0) : 0; };
    const kurt = a => { const m = mean(a), s = std(a), n = a.length; return s && n > 3 ? a.reduce((acc, v) => acc + ((v - m) / s) ** 4, 0) / n - 3 : 0; };
    const pctile = (a, p) => { const s = [...a].sort((x, y) => x - y), i = (p / 100) * (s.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return lo === hi ? s[lo] : s[lo] * (hi - i) + s[hi] * (i - lo); };
    function histo(a, bins) {
      if (!a.length) return { edges: [], counts: [] };
      const mn = Math.min(...a), mx = Math.max(...a), w = (mx - mn || 1) / bins;
      const edges = Array.from({ length: bins + 1 }, (_, i) => mn + i * w);
      const counts = new Array(bins).fill(0);
      for (const v of a) { let i = Math.floor((v - mn) / w); if (i >= bins) i = bins - 1; if (i < 0) i = 0; counts[i]++; }
      return { edges, counts };
    }
