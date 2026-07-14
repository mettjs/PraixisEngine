// Usage view — token and request totals per app.
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

    totalUsageTokens()   { return this.usage.reduce((s, a) => s + (a.total_tokens || 0), 0); },
    totalUsageRequests() { return this.usage.reduce((s, a) => s + (a.requests     || 0), 0); },

  };
}
