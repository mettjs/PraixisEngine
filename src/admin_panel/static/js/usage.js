// Usage view — token and request totals per app, expandable per model.
function _adminUsage() {
  return {

    async loadUsage() {
      this.loading.usage = true;
      try {
        const r = await this.req('GET', '/api/system/usage');
        if (r.ok) {
          const d          = await r.json();
          this.usage       = (d.apps || []).sort((a, b) => b.total_tokens - a.total_tokens);
          this.usageLoaded = true;
        } else if (r.status !== 401) {
          this.showToast('Failed to load usage data.', 'error');
        }
      } catch {
        this.showToast('Failed to load usage data — network error.', 'error');
      } finally {
        this.loading.usage = false;
      }
    },

    /**
     * Expand an app row to its per-model split. The all-apps endpoint stays a
     * single cheap call; the breakdown is fetched per app, only when asked for,
     * and cached for the life of the view.
     */
    async toggleAppModels(appName) {
      if (this.expandedApp === appName) { this.expandedApp = null; return; }
      this.expandedApp = appName;
      if (this.usageModels[appName]) return;
      try {
        const r = await this.req('GET', '/api/system/usage/' + encodeURIComponent(appName));
        if (r.ok) this.usageModels[appName] = (await r.json()).by_model || [];
      } catch {
        this.showToast('Failed to load the per-model breakdown.', 'error');
      }
    },

    appModels(appName) { return this.usageModels[appName] || []; },

    totalUsageTokens()   { return this.usage.reduce((s, a) => s + (a.total_tokens || 0), 0); },
    totalUsageRequests() { return this.usage.reduce((s, a) => s + (a.requests     || 0), 0); },

  };
}
