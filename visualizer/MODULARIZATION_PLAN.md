# Analyzer Modularization (Implemented)

The analyzer is now split into maintainable modules while preserving `analyzer/index.html` as the single entrypoint.

## Access

- Open `analyzer/index.html` directly in your browser (same as before).
- Monolith rollback copy is stored at:
  - `analyzer/storage/index.monolith.backup.20260414_201852.html`

## Implemented Structure

```text
analyzer/
  index.html
  styles/
    base.css
    layout.css
    controls.css
    tabs.css
    charts.css
    modals.css
  js/
    core/
      config.js
      state.js
      stats.js
      helpers.js
      fair_value.js
    render/
      canvas.js
      primitives.js
      chart_main.js
      chart_pos_pnl.js
      tooltip_log.js
    analytics/
      spread.js
      returns.js
      microstructure.js
      mm_optimizer.js
      strategy.js
      signals.js
      dynamics.js
      export.js
    data/
      prices_parser.js
      trades_parser.js
      logs_parser.js
      official_json_parser.js
      loaders.js
    ui/
      trade_log.js
      quick_stats.js
      tabs.js
      controls.js
    bootstrap.js
```

## Section-to-File Map

- Section 1 -> `js/core/config.js`
- Section 2 -> `js/core/state.js`
- Section 3, 8 -> `js/render/canvas.js`
- Section 4 -> `js/core/fair_value.js`
- Section 5, 7 -> `js/core/helpers.js`
- Section 6 -> `js/core/stats.js`
- Section 9 -> `js/render/primitives.js`
- Sections 10-12 -> `js/render/chart_main.js`
- Section 13 -> `js/render/chart_pos_pnl.js`
- Sections 14-15 -> `js/render/tooltip_log.js`
- Sections 16-17 -> `js/ui/trade_log.js`, `js/ui/quick_stats.js`
- Sections 18-22 -> `js/analytics/*`
- Sections 23-24 -> `js/ui/tabs.js`, `js/ui/controls.js`
- Section 25 -> `js/data/prices_parser.js`, `js/data/trades_parser.js`, `js/data/logs_parser.js`, `js/data/official_json_parser.js`
- Section 26 -> `js/data/loaders.js`
- Section 27 -> `js/bootstrap.js`

## Guardrails In Place

- `index.html` remains stable and only references external CSS/JS modules.
- Backup copy in `analyzer/storage/` enables instant rollback.
- All referenced script files are existence-checked.
- All JS modules pass `node --check` syntax validation.
- Parser failures are non-fatal in batch loading (`console.warn` + continue).
- Fair-value method switching remains centralized in `core/fair_value.js`.

## Extending Safely

### Add a new fair-value model

1. Add model ID in `js/core/config.js`.
2. Add compute branch in `js/core/fair_value.js`.
3. Add selector option in analyzer controls markup in `index.html`.
4. Ensure export metadata in `js/analytics/export.js` reflects the model.

### Add a new analytics component/tab

1. Implement logic in a new `js/analytics/<name>.js`.
2. Wire render call in tab-control flow (`js/ui/tabs.js`).
3. Add tab button/content container in `index.html` sidebar.

### Add parser enhancements

1. Update relevant `js/data/*_parser.js`.
2. Keep snapshot/trade schema compatibility.
3. Keep failures non-fatal and logged as warnings.

