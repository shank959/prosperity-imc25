// =====================================================================
    // SECTION 23 — TAB SWITCHING
    // =====================================================================
    document.querySelectorAll('.tab-bar button').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-bar button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
        const tc = document.querySelector(`.tab-content[data-tab="${btn.dataset.tab}"]`);
        if (tc) {
          tc.classList.add('active');
          if (btn.dataset.tab === 'spread') renderSpreadTab();
          if (btn.dataset.tab === 'returns') renderReturnsTab();
          if (btn.dataset.tab === 'bots') renderBotsTab();
          if (btn.dataset.tab === 'mm') renderMMTab();
          if (btn.dataset.tab === 'strategy') renderStrategyTab();
          if (btn.dataset.tab === 'signals') renderSignalsTab();
          if (btn.dataset.tab === 'dynamics') renderDynamicsTab();
          if (btn.dataset.tab === 'botIntel') renderBotIntelTab();
        }
      });
    });
