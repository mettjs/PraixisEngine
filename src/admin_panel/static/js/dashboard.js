// Dashboard view — health checks, system stats, GPU status.
function _adminDashboard() {
  return {

    async loadDashboard() {
      this.loading.dashboard = true;
      await Promise.all([this.loadStats(), this.loadGpu(), this.loadRestartState()]);
      this.loading.dashboard = false;
      this.dashboardLoaded   = true;
      this.loadHealth();
    },

    async loadHealth() {
      this.health.redis    = null;
      this.health.vectordb = null;
      this.health.llm      = null;
      this.llmBackends = [];
      const load = async (svc) => {
        try {
          const r = await this.req('GET', '/api/system/health/' + svc);
          if (!r.ok) { this.health[svc] = 'offline'; return; }
          const body = await r.json();
          this.health[svc] = body.status;
          // /health/llm pings each backend once and names the models on it.
          if (svc === 'llm') this.llmBackends = body.backends || [];
        } catch { this.health[svc] = 'offline'; }
      };
      await Promise.all([load('redis'), load('vectordb'), load('llm')]);
    },

    /**
     * Whether models.yaml has drifted from what this process is serving.
     * Fetched here too so the sidebar marker can appear without first visiting
     * the Models view — the operator who has not been there is the one who
     * needs telling.
     */
    async loadRestartState() {
      try {
        const r = await this.req('GET', '/api/system/models');
        if (r.ok) this.restartRequired = (await r.json()).restart_required;
      } catch { /* the marker just stays hidden */ }
    },

    async loadStats() {
      try {
        const r = await this.req('GET', '/api/system/stats');
        if (r.ok) this.stats = await r.json();
      } catch {}
    },

    async loadGpu() {
      try {
        const r = await this.req('GET', '/api/system/gpu');
        if (r.ok) this.gpu = await r.json();
      } catch {}
    },

    // Both bar helpers work off any pool's figures; passing none reads the
    // top-level (default pool) numbers the payload has always carried.
    gpuBarWidth(pool = null) {
      const p = pool || this.gpu;
      if (!p || !p.slots_total) return '0%';
      return Math.min(100, Math.round((p.slots_in_use / p.slots_total) * 100)) + '%';
    },

    gpuBarColor(pool = null) {
      const p = pool || this.gpu;
      if (!p || !p.slots_total) return 'bg-slate-600';
      const pct = p.slots_in_use / p.slots_total;
      if (pct < 0.5)  return 'bg-green-500';
      if (pct < 0.85) return 'bg-amber-500';
      return 'bg-red-500';
    },

    /** Pools as a sorted list, so the template can iterate them. */
    gpuPools() {
      return Object.entries(this.gpu.pools || {})
        .map(([name, stats]) => ({ name, ...stats }))
        .sort((a, b) => a.name.localeCompare(b.name));
    },

    /** True once there is more than one backend's capacity to show. */
    hasMultiplePools() {
      return this.gpuPools().length > 1;
    },

  };
}
