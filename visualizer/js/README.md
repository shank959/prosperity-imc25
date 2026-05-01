# Analyzer JS Modules

`index.html` now loads analyzer logic from this directory instead of inline monolith JS.

## Runtime Load Order (required)

1. `core/config.js`
2. `core/state.js`
3. `core/stats.js`
4. `core/helpers.js`
5. `core/fair_value.js`
6. `render/canvas.js`
7. `render/primitives.js`
8. `render/chart_main.js`
9. `render/chart_pos_pnl.js`
10. `render/tooltip_log.js`
11. `analytics/spread.js`
12. `analytics/returns.js`
13. `analytics/microstructure.js`
14. `analytics/mm_optimizer.js`
15. `analytics/strategy.js`
16. `analytics/signals.js`
17. `analytics/dynamics.js`
18. `analytics/export.js`
19. `ui/trade_log.js`
20. `ui/quick_stats.js`
21. `ui/tabs.js`
22. `ui/controls.js`
23. `data/prices_parser.js`
24. `data/trades_parser.js`
25. `data/logs_parser.js`
26. `data/official_json_parser.js`
27. `data/loaders.js`
28. `bootstrap.js`

## Guardrails

- Keep top-level identifiers unique (scripts share a global scope).
- Do not silently change snapshot/trade object shape.
- Parser-level failures should warn and continue for multi-file loads.
- Keep fair-value computation centralized in `core/fair_value.js`.

## Extension Quickstart

### New fair value model

1. Add model constant in `core/config.js`.
2. Add implementation path in `core/fair_value.js`.
3. Add dropdown option in `index.html`.
4. Confirm export metadata in `analytics/export.js`.

### New UI control

1. Add control markup in `index.html`.
2. Wire behavior in `ui/controls.js`.
3. Trigger `scheduleRender()` and cache invalidation where needed.

### New parser capability

1. Update relevant file under `data/`.
2. Keep non-fatal error handling in `data/loaders.js`.
3. Validate with mixed multi-day and official JSON inputs.

