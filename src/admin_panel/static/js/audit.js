// Audit Log view — paginated event feed with app filter.
function _adminAudit() {
  return {

    async loadAudit(append = false) {
      this.loading.audit = true;
      try {
        const params = { limit: this.auditLimit, offset: this.auditOffset };
        const path   = this.auditAppFilter
          ? '/api/system/audit/' + encodeURIComponent(this.auditAppFilter)
          : '/api/system/audit';
        const r = await this.req('GET', path, params);
        if (r.ok) {
          const d           = await r.json();
          const events      = d.events || [];
          this.auditEvents  = append ? [...this.auditEvents, ...events] : events;
          this.auditHasMore = events.length >= this.auditLimit;
          this.auditLoaded  = true;
        } else if (r.status !== 401) {
          this.showToast('Failed to load audit log.', 'error');
        }
      } catch {
        this.showToast('Failed to load audit log — network error.', 'error');
      } finally {
        this.loading.audit = false;
      }
    },

    async loadMoreAudit() {
      this.auditOffset += this.auditLimit;
      await this.loadAudit(true);
    },

    onAuditFilterChange() {
      clearTimeout(this.auditFilterTimer);
      this.auditFilterTimer = setTimeout(async () => {
        this.auditOffset = 0;
        this.auditEvents = [];
        this.auditLoaded = false;
        await this.loadAudit();
      }, 400);
    },

  };
}
