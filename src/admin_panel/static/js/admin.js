// adminApp — core state, auth, navigation, and module assembly.
// Must be loaded after all _admin* module files and before Alpine.js.
function adminApp() {

  const core = {

    // ── Auth ───────────────────────────────────────────────────────────────────
    initializing:  true,
    isLoggedIn:    false,
    loggedInUser:  '',
    loginUsername: '',
    loginPassword: '',
    loginError:    '',
    loginLoading:  false,
    authHeader:    '',

    // ── Navigation ─────────────────────────────────────────────────────────────
    view: 'dashboard',

    get viewTitle() {
      const titles = { dashboard: 'Dashboard', keys: 'API Keys', models: 'Models', usage: 'Usage & Tokens', audit: 'Audit Log', vector: 'Vector DB' };
      return titles[this.view] ?? '';
    },

    get isLoading() {
      return this.loading.dashboard || this.loading.keys || this.loading.usage || this.loading.audit || this.loading.vector;
    },

    // ── Loading flags ──────────────────────────────────────────────────────────
    loading: { dashboard: false, keys: false, models: false, usage: false, audit: false, vector: false },

    // ── Data stores ────────────────────────────────────────────────────────────
    health: { api: 'online', redis: null, vectordb: null, llm: null },
    stats:  {},
    gpu:    {},
    registry:        { models: [], default: null, roles: {}, file: null },
    draft:           { models: [], roles: {}, pools: {} },
    draftError:      '',
    modelsLoaded:    false,
    restartRequired: false,
    llmBackends:   [],
    usageModels:   {},   // app_name -> per-model rows, fetched on expand
    expandedApp:   null,
    keys:   [],
    usage:  [],

    auditEvents:      [],
    auditOffset:      0,
    auditLimit:       50,
    auditHasMore:     false,
    auditAppFilter:   '',
    auditLoaded:      false,
    auditFilterTimer: null,

    dashboardLoaded: false,
    keysLoaded:      false,
    usageLoaded:     false,

    // ── Vector DB ──────────────────────────────────────────────────────────────
    vectorCollections:  [],
    vectorLoaded:       false,
    vectorExpanded:     null,
    vectorFiles:        {},
    vectorFilesLoading: {},

    vectorSearch: {
      query:      '',
      appName:    '',
      collection: '',
      nResults:   5,
      loading:    false,
      done:       false,
      results:    [],
      scoreType:  'rrf',
      expanded:   {},
    },

    // ── Modal ──────────────────────────────────────────────────────────────────
    modal:           null,
    modalData:       {},
    newAppName:      '',
    newAppNameError: '',
    newKeyModels:        [],    // empty = unrestricted
    newKeyDefaultModel:  '',
    modalLoading:    false,

    // ── Toast ──────────────────────────────────────────────────────────────────
    toast:       { visible: false, message: '', type: 'success' },
    _toastTimer: null,

    // ── Auto-refresh handles ───────────────────────────────────────────────────
    _gpuTimer:    null,
    _healthTimer: null,

    // ══════════════════════════════════════════════════════════════════════════
    // INIT
    // ══════════════════════════════════════════════════════════════════════════
    async init() {
      const token = sessionStorage.getItem('praixis_admin_token');
      const user  = sessionStorage.getItem('praixis_admin_user');
      if (!token) { this.initializing = false; return; }
      this.authHeader = token;
      try {
        const result = await this._verifyAuth();
        if (result === true) {
          this.loggedInUser = user || 'Admin';
          this.isLoggedIn   = true;
          this.startAutoRefresh();
          this.loadDashboard();
        } else if (result === 'auth') {
          this.clearSession();
        }
        // false (server error) or network throw: preserve token, show login
      } catch { /* network error — token preserved, user can retry */ }
      finally { this.initializing = false; }
    },

    clearSession() {
      sessionStorage.removeItem('praixis_admin_token');
      sessionStorage.removeItem('praixis_admin_user');
      this.authHeader = '';
    },

    // Returns true on success, 'auth' on 401/403, false on any other server error.
    // Throws on network failure.
    async _verifyAuth() {
      const r = await fetch('/api/system/auth/verify', {
        headers: { Authorization: 'Basic ' + this.authHeader },
      });
      if (r.status === 401 || r.status === 403) return 'auth';
      return r.ok ? true : false;
    },

    // ══════════════════════════════════════════════════════════════════════════
    // AUTH
    // ══════════════════════════════════════════════════════════════════════════
    async login() {
      this.loginError   = '';
      this.loginLoading = true;
      try {
        this.authHeader = btoa(this.loginUsername + ':' + this.loginPassword);
        const result    = await this._verifyAuth();
        if (result === true) {
          this.loggedInUser  = this.loginUsername;
          sessionStorage.setItem('praixis_admin_token', this.authHeader);
          sessionStorage.setItem('praixis_admin_user',  this.loginUsername);
          this.isLoggedIn    = true;
          this.loginPassword = '';
          this.startAutoRefresh();
          this.loadDashboard();
        } else {
          this.loginError = result === 'auth'
            ? 'Invalid credentials. Please try again.'
            : 'Server error. Please try again later.';
          this.authHeader = '';
        }
      } catch {
        this.loginError = 'Connection error. Is the server running?';
        this.authHeader = '';
      } finally {
        this.loginLoading = false;
      }
    },

    logout() {
      this.clearSession();
      this.isLoggedIn         = false;
      this.modal              = null;
      this.modalData          = {};
      this.loginUsername      = '';
      this.view               = 'dashboard';
      this.dashboardLoaded    = false;
      this.keysLoaded         = false;
      this.usageLoaded        = false;
      this.modelsLoaded       = false;
      this.auditLoaded        = false;
      this.auditEvents        = [];
      this.vectorCollections  = [];
      this.vectorLoaded       = false;
      this.vectorExpanded     = null;
      this.vectorFiles        = {};
      this.vectorFilesLoading = {};
      this.vectorSearch       = { query: '', appName: '', collection: '', nResults: 5, loading: false, done: false, results: [], scoreType: 'rrf', expanded: {} };
      // Everything below holds another operator's data — the registry payload
      // includes plaintext api_key values, and the usage cache their token
      // counts. On a shared machine both would render before the next admin's
      // first fetch returns.
      this.keys               = [];
      this.usage              = [];
      this.usageModels        = {};
      this.expandedApp        = null;
      this.registry           = { models: [], default: null, roles: {}, file: null };
      this.draft              = { models: [], roles: {}, pools: {} };
      this.draftError         = '';
      this.restartRequired    = false;
      this.llmBackends        = [];
      this.stats              = {};
      this.gpu                = {};
      this.stopAutoRefresh();
    },

    // ══════════════════════════════════════════════════════════════════════════
    // HTTP HELPER
    // ══════════════════════════════════════════════════════════════════════════
    async req(method, path, params = null, jsonBody = undefined) {
      // Array values become repeated params (?models=a&models=b), which is what
      // FastAPI expects for a list query field; null/'' values are dropped.
      const query = params
        ? new URLSearchParams(
            Object.entries(params).flatMap(([k, v]) =>
              Array.isArray(v) ? v.map((item) => [k, item])
                : (v === null || v === undefined || v === '' ? [] : [[k, v]])
            )
          ).toString()
        : '';
      const url = query ? path + '?' + query : path;
      const init = { method, headers: { Authorization: 'Basic ' + this.authHeader } };
      if (jsonBody !== undefined) {
        // `null` is a meaningful body here (it deletes models.yaml), so the
        // guard is on the argument being passed at all, not on its value.
        init.headers['Content-Type'] = 'application/json';
        init.body = JSON.stringify(jsonBody);
      }
      const r = await fetch(url, init);
      if (r.status === 401 && this.isLoggedIn) {
        // Credentials changed server-side (e.g. restart with new admin creds):
        // drop to the login screen instead of failing every call silently.
        this.logout();
        this.loginError = 'Session expired — please sign in again.';
      }
      return r;
    },

    // ══════════════════════════════════════════════════════════════════════════
    // NAVIGATION & REFRESH
    // ══════════════════════════════════════════════════════════════════════════
    async navigate(v) {
      this.view = v;
      if      (v === 'dashboard' && !this.dashboardLoaded) await this.loadDashboard();
      else if (v === 'keys'      && !this.keysLoaded)      await this.loadKeys();
      else if (v === 'models'    && !this.modelsLoaded)    await this.loadModels();
      else if (v === 'usage'     && !this.usageLoaded)     await this.loadUsage();
      else if (v === 'vector'    && !this.vectorLoaded)    await this.loadVectorCollections();
      else if (v === 'audit'     && !this.auditLoaded) {
        this.auditOffset = 0;
        this.auditEvents = [];
        await this.loadAudit();
      }
    },

    async refreshCurrentView() {
      if      (this.view === 'dashboard') { this.dashboardLoaded = false; await this.loadDashboard(); }
      else if (this.view === 'keys')      await this.loadKeys();
      else if (this.view === 'models')    await this.loadModels();
      else if (this.view === 'usage')     {
        // The per-model rows are cached until asked for again, so a refresh
        // that kept them would show them under totals they no longer sum to.
        this.usageModels = {};
        this.expandedApp = null;
        await this.loadUsage();
      }
      else if (this.view === 'vector')    {
        // Collapse any open row too — a row expanded over an empty file cache
        // would render blank (neither loading nor list).
        this.vectorFiles        = {};
        this.vectorFilesLoading = {};
        this.vectorExpanded     = null;
        await this.loadVectorCollections();
      }
      else if (this.view === 'audit')     { this.auditOffset = 0; this.auditEvents = []; await this.loadAudit(); }
    },

    startAutoRefresh() {
      this.stopAutoRefresh();  // never stack a second pair of timers
      this._gpuTimer    = setInterval(() => { if (this.view === 'dashboard') this.loadGpu(); },    10000);
      this._healthTimer = setInterval(() => { if (this.view === 'dashboard') this.loadHealth(); }, 30000);
    },

    stopAutoRefresh() {
      clearInterval(this._gpuTimer);
      clearInterval(this._healthTimer);
    },

  };

  // Merge feature modules into core, preserving getter descriptors.
  [
    _adminDashboard(),
    _adminKeys(),
    _adminModels(),
    _adminUsage(),
    _adminAudit(),
    _adminVector(),
    _adminHelpers(),
  ].forEach(mod => Object.defineProperties(core, Object.getOwnPropertyDescriptors(mod)));

  return core;
}
