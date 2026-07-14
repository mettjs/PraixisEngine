// API Keys view — generate, rotate, revoke, wipe sessions, reset GPU slots.
function _adminKeys() {
  return {

    async loadKeys() {
      this.loading.keys = true;
      try {
        const r = await this.req('GET', '/api/system/keys');
        if (r.ok) {
          const d         = await r.json();
          this.keys       = (d.keys || []).sort((a, b) => a.app_name.localeCompare(b.app_name));
          this.keysLoaded = true;
        } else if (r.status !== 401) {
          this.showToast('Failed to load API keys.', 'error');
        }
      } catch {
        this.showToast('Failed to load API keys — network error.', 'error');
      } finally {
        this.loading.keys = false;
      }
    },

    openGenerateKeyModal() {
      this.newAppName      = '';
      this.newAppNameError = '';
      this.modalLoading    = false;
      this.modal           = 'generateKey';
    },

    async generateKey() {
      this.newAppNameError = '';
      if (!/^[a-zA-Z0-9_-]{3,63}$/.test(this.newAppName)) {
        this.newAppNameError = 'Must be 3–63 characters: letters, numbers, _ or -';
        return;
      }
      this.modalLoading = true;
      try {
        const r = await this.req('POST', '/api/system/keys/generate', { app_name: this.newAppName });
        if (r.ok) {
          this.modalData = await r.json();
          this.modal     = 'newKeyResult';
          await this.loadKeys();
        } else {
          const d = await r.json().catch(() => ({}));
          this.newAppNameError = d.detail || 'Failed to generate key.';
        }
      } finally {
        this.modalLoading = false;
      }
    },

    async copyKey() {
      try {
        await navigator.clipboard.writeText(this.modalData.api_key);
        this.showToast('API key copied to clipboard!', 'success');
      } catch {
        this.showToast('Auto-copy failed — select the key above and copy manually.', 'error');
      }
    },

    openRotateModal(key) {
      this.modalData    = key;
      this.modalLoading = false;
      this.modal        = 'rotateKey';
    },

    async rotateKey() {
      this.modalLoading = true;
      try {
        const r = await this.req('POST', '/api/system/keys/rotate', { key_hash: this.modalData.key_hash });
        if (r.ok) {
          this.modalData = await r.json();  // shows in the newKeyResult modal
          this.modal     = 'newKeyResult';
          await this.loadKeys();
        } else {
          const d = await r.json().catch(() => ({}));
          this.showToast(d.detail || 'Rotate failed.', 'error');
          this.modal = null;
        }
      } catch {
        this.showToast('Rotate failed — network error.', 'error');
        this.modal = null;
      } finally {
        this.modalLoading = false;
      }
    },

    openRevokeModal(key) {
      this.modalData    = key;
      this.modalLoading = false;
      this.modal        = 'revokeKey';
    },

    async revokeKey() {
      this.modalLoading = true;
      try {
        const r = await this.req('DELETE', '/api/system/keys/revoke-by-hash', { key_hash: this.modalData.key_hash });
        if (r.ok) {
          this.modal = null;
          this.showToast('Key for "' + this.modalData.app_name + '" revoked.', 'success');
          await this.loadKeys();
        } else {
          const d = await r.json().catch(() => ({}));
          this.showToast(d.detail || 'Revoke failed.', 'error');
          this.modal = null;
        }
      } finally {
        this.modalLoading = false;
      }
    },

    openWipeSessionsModal(key) {
      this.modalData    = key;
      this.modalLoading = false;
      this.modal        = 'wipeSessions';
    },

    async wipeSessions() {
      this.modalLoading = true;
      try {
        const r = await this.req('DELETE', '/api/system/sessions/' + encodeURIComponent(this.modalData.app_name));
        if (r.ok) {
          const d    = await r.json();
          this.modal = null;
          this.showToast(d.sessions_deleted + ' session(s) wiped for "' + this.modalData.app_name + '".', 'success');
          if (this.view === 'dashboard') await this.loadStats();
        } else {
          this.showToast('Failed to wipe sessions.', 'error');
          this.modal = null;
        }
      } finally {
        this.modalLoading = false;
      }
    },

    openResetGpuModal() {
      this.modalLoading = false;
      this.modal        = 'resetGpu';
    },

    async resetGpu() {
      this.modalLoading = true;
      try {
        const r = await this.req('POST', '/api/system/gpu/reset');
        if (r.ok) {
          this.modal = null;
          await this.loadGpu();
          this.showToast('GPU slot counter reset to 0.', 'success');
        } else {
          this.showToast('Failed to reset GPU counter.', 'error');
          this.modal = null;
        }
      } finally {
        this.modalLoading = false;
      }
    },

  };
}
