// SECTION 26 — FILE LOADING

function setLoading(msg) {
  const el = document.getElementById('loading');
  const sub = document.getElementById('loadingSub');
  if (msg) { el.classList.add('show'); sub.textContent = msg; }
  else el.classList.remove('show');
}

function afterLoad() {
  const pSel = document.getElementById('productSelect');
  const existing = new Set([...pSel.options].map(o => o.value));
  for (const p of Store.products) {
    if (!existing.has(p)) { pSel.add(new Option(p, p)); }
  }
  const dSel = document.getElementById('daySelect');
  const existD = new Set([...dSel.options].map(o => o.value));
  for (const d of Store.days) {
    if (!existD.has(d.toString())) { dSel.add(new Option('Day ' + d, d)); }
  }
  if (!currentProduct && Store.products.length) {
    currentProduct = Store.products[0];
    pSel.value = currentProduct;
  }

  // Populate trader name filter (local output.log only)
  const traderSel = document.getElementById('filterTrader');
  if (traderSel) {
    const existingNames = new Set([...traderSel.options].map(o => o.value).filter(v => v));
    for (const name of Store.traderNames) {
      if (!existingNames.has(name)) traderSel.add(new Option(name, name));
    }
  }

  // Pre-compute daily price ranges per product (used by classifyTrade 'I' detection)
  Store.dailyRanges = {};
  for (const product of Store.products) {
    Store.dailyRanges[product] = {};
    for (const snap of (Store.data[product] || [])) {
      if (snap.mid == null) continue;
      if (!Store.dailyRanges[product][snap.day]) {
        Store.dailyRanges[product][snap.day] = { min: Infinity, max: -Infinity };
      }
      const dr = Store.dailyRanges[product][snap.day];
      if (snap.mid < dr.min) dr.min = snap.mid;
      if (snap.mid > dr.max) dr.max = snap.mid;
    }
  }

  // Compute MTM PnL after all data (prices + trades + logs) is loaded.
  // Must run after recomputeAllFairValues (already called inside parsePricesCSV).
  recomputeAllPnl();

  initCanvases();
  autoFitViewport();
  scheduleRender();
}

async function loadMultipleFiles(files, type) {
  // Sort by filename so that day_-2 is loaded before day_-1 (ensures consistent offsets)
  const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name));
  setLoading(`Loading 0/${sorted.length}...`);
  for (let i = 0; i < sorted.length; i++) {
    const file = sorted[i];
    setLoading(`Parsing ${i + 1}/${sorted.length}: ${file.name}`);
    await new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = ev => {
        try {
          if (type === 'prices') parsePricesCSV(ev.target.result);
          else if (type === 'trades') parseTradesCSV(ev.target.result, file.name);
          else if (type === 'log') parseOutputLog(ev.target.result);
          else if (type === 'official') parseIMCOfficialJSON(ev.target.result);
        } catch (e) {
          console.warn(`Skipping malformed file ${file.name}: ${e.message}`);
        }
        resolve();
      };
      reader.onerror = () => resolve();
      reader.readAsText(file);
    });
  }
  afterLoad();
  setLoading(null);
}

document.getElementById('logFile').addEventListener('change', e => {
  if (e.target.files[0]) loadMultipleFiles([e.target.files[0]], 'log');
});
document.getElementById('pricesFile').addEventListener('change', e => {
  if (e.target.files.length) loadMultipleFiles([...e.target.files], 'prices');
});
document.getElementById('tradesFile').addEventListener('change', e => {
  if (e.target.files.length) loadMultipleFiles([...e.target.files], 'trades');
});
document.getElementById('officialJsonFile').addEventListener('change', e => {
  if (e.target.files[0]) loadMultipleFiles([e.target.files[0]], 'official');
});

// Drag and drop
const dropZone = document.getElementById('dropZone');
document.body.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('show');
});
document.body.addEventListener('dragleave', e => {
  if (!e.relatedTarget || !document.body.contains(e.relatedTarget)) {
    dropZone.classList.remove('show');
  }
});
document.body.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('show');
  const files = [...e.dataTransfer.files];
  if (!files.length) return;

  // Classify by first file's name; all dropped files treated as same type
  const name = files[0].name.toLowerCase();
  const baseName = name.replace(/\.[^.]+$/, '');
  let type = 'prices';

  // Numeric filename (e.g. 72288.log, 72288.json) = official IMC site download
  if (name.endsWith('.json') || (name.endsWith('.log') && /^\d+$/.test(baseName))) type = 'official';
  else if (name.endsWith('.log') || name.endsWith('.txt')) type = 'log';
  else if (name.includes('trade')) type = 'trades';
  loadMultipleFiles(files, type);
});
