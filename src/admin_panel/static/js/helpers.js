// Shared UI helpers — toast, status colours, audit formatting.
function _adminHelpers() {
  return {

    showToast(message, type = 'success') {
      clearTimeout(this._toastTimer);
      this.toast       = { visible: true, message, type };
      this._toastTimer = setTimeout(() => { this.toast.visible = false; }, 3500);
    },

    statusColor(status) {
      if (status === 'online')  return 'text-green-400';
      if (status === 'offline') return 'text-red-400';
      return 'text-slate-500';
    },

    statusDotColor(status) {
      if (status === 'online')  return 'bg-green-400 shadow-[0_0_6px_rgba(74,222,128,0.6)]';
      if (status === 'offline') return 'bg-red-400';
      return 'bg-slate-500 animate-pulse';
    },

    actionBadge(action) {
      const map = {
        KEY_GENERATED: 'bg-green-400/15  text-green-400  ring-green-500/20',
        KEY_ROTATED:   'bg-blue-400/15   text-blue-400   ring-blue-500/20',
        KEY_REVOKED:   'bg-red-400/15    text-red-400    ring-red-500/20',
        AUTH_FAIL:     'bg-red-400/15    text-red-400    ring-red-500/20',
        SESSION_WIPED: 'bg-amber-400/15  text-amber-400  ring-amber-500/20',
        GPU_RESET:     'bg-blue-400/15   text-blue-400   ring-blue-500/20',
      };
      return map[action] ?? 'bg-slate-400/15 text-slate-400 ring-slate-500/20';
    },

    formatDate(iso) {
      if (!iso) return '—';
      try { return new Date(iso).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'medium' }); }
      catch { return iso; }
    },

    formatDetails(details) {
      if (!details) return '—';
      if (typeof details === 'string') return details;
      return Object.entries(details)
        .map(([k, v]) => k + ': ' + (v !== null && typeof v === 'object' ? JSON.stringify(v) : v))
        .join(' · ');
    },

  };
}
