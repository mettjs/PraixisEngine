// Models view — read the registry, edit models.yaml, restart to apply.
//
// The engine parses models.yaml once at startup and derives its GPU pools from
// it, so saving here writes the file and nothing more: `restartRequired` is how
// the UI stays honest about the gap between what is saved and what is serving.
function _adminModels() {
  return {

    async loadModels() {
      this.loading.models = true;
      try {
        const r = await this.req('GET', '/api/system/models');
        if (r.ok) {
          this.registry = await r.json();
          this.restartRequired = this.registry.restart_required;
          this.modelsLoaded = true;
          this._seedDraft();
        } else if (r.status !== 401) {
          this.showToast('Failed to load the model registry.', 'error');
        }
      } catch {
        this.showToast('Failed to load the model registry — network error.', 'error');
      } finally {
        this.loading.models = false;
      }
    },

    /**
     * The editable draft is the file as *written*, not the resolved registry:
     * round-tripping the resolved form would bake every env-var default into
     * the file and silently freeze values the author left open.
     *
     * With no file yet, seed it from what is running so that adding a second
     * model never drops the first — the single biggest way to break a
     * deployment when moving off the env-var fallback.
     */
    _seedDraft() {
      // A file that could not be read is NOT an absent one: seeding from env
      // and saving would destroy something recoverable.
      if (this.registry.file_error) {
        this.draft = { models: [], roles: {}, pools: {} };
        this.draftError = '';
        return;
      }
      const file = this.registry.file;
      if (file && Array.isArray(file.models)) {
        this.draft = JSON.parse(JSON.stringify(file));
      } else {
        this.draft = {
          default: this.registry.default,
          roles: { ...this.registry.roles },
          pools: {},
          models: this.registry.models.map((m) => ({ id: m.id, model: m.model })),
        };
      }
      this.draft.models ||= [];
      this.draft.roles ||= {};
      this.draft.pools ||= {};
      this.draftError = '';
    },

    hasRegistryFile() {
      return !!(this.registry.file && Array.isArray(this.registry.file.models));
    },

    /** True when models.yaml exists but could not be read or parsed. */
    hasBrokenRegistryFile() {
      return !!this.registry.file_error;
    },

    /** Whether a save could succeed at all — the file is often mounted :ro. */
    canSave() {
      return this.registry.writable !== false && !this.hasBrokenRegistryFile();
    },

    addDraftModel() {
      this.draft.models.push({ id: '', model: '' });
    },

    removeDraftModel(index) {
      const [removed] = this.draft.models.splice(index, 1);
      // Dangling references are a startup error, so clear them here instead of
      // letting the save fail with a message about a model that is now gone.
      if (this.draft.default === removed.id) this.draft.default = this.draft.models[0]?.id || '';
      for (const role of Object.keys(this.draft.roles)) {
        if (this.draft.roles[role] === removed.id) delete this.draft.roles[role];
      }
    },

    /**
     * Strip empty optional fields so the file stays as short as it was typed.
     *
     * Each entry starts as a copy of what was loaded, so fields this form does
     * not render — `params` above all — survive an edit instead of being
     * dropped by a whitelist. Only the fields the form owns are normalised.
     */
    _cleanDraft() {
      const FORM_FIELDS = ['api_url', 'api_key', 'context_window', 'pool'];
      const clean = { models: [] };
      for (const m of this.draft.models) {
        const entry = { ...m, id: (m.id || '').trim(), model: (m.model || '').trim() };
        for (const field of FORM_FIELDS) {
          if (!entry[field]) delete entry[field];
          else if (typeof entry[field] === 'string') entry[field] = entry[field].trim();
        }
        if (entry.context_window) entry.context_window = Number(entry.context_window);
        clean.models.push(entry);
      }
      if (this.draft.default) clean.default = this.draft.default;
      const roles = Object.fromEntries(Object.entries(this.draft.roles).filter(([, v]) => v));
      if (Object.keys(roles).length) clean.roles = roles;
      const pools = Object.fromEntries(
        Object.entries(this.draft.pools).filter(([k, v]) => k && v).map(([k, v]) => [k, Number(v)]),
      );
      if (Object.keys(pools).length) clean.pools = pools;
      // `default` must come first in the written file for readability; the
      // server re-validates whatever arrives regardless.
      return { ...(clean.default && { default: clean.default }), ...clean };
    },

    async saveModels() {
      this.draftError = '';
      if (!this.canSave()) {
        this.draftError = this.hasBrokenRegistryFile()
          ? 'models.yaml exists but could not be read — fix it on disk before saving over it.'
          : 'models.yaml is not writable by the engine (mounted read-only?).';
        return;
      }
      this.modalLoading = true;
      try {
        const r = await this.req('PUT', '/api/system/models', null, this._cleanDraft());
        const body = await r.json().catch(() => ({}));
        if (r.ok) {
          this.restartRequired = body.restart_required;
          this.showToast('Saved. Restart the engine to apply.', 'success');
          await this.loadModels();
        } else {
          // The engine validates the whole document, so this is the same
          // message a bad file would abort startup with.
          this.draftError = body.detail || 'Save failed.';
        }
      } catch {
        this.draftError = 'Save failed — network error.';
      } finally {
        this.modalLoading = false;
      }
    },

    openRemoveRegistryModal() {
      this.modalLoading = false;
      this.modal = 'removeRegistry';
    },

    /** Delete models.yaml and fall back to the single env-var model. */
    async removeRegistry() {
      this.modalLoading = true;
      try {
        const r = await this.req('DELETE', '/api/system/models');
        if (r.ok) {
          this.modal = null;
          this.showToast('models.yaml removed. Restart to fall back to MODEL_NAME.', 'success');
          await this.loadModels();
        } else {
          this.showToast('Could not remove models.yaml.', 'error');
          this.modal = null;
        }
      } finally {
        this.modalLoading = false;
      }
    },

  };
}
