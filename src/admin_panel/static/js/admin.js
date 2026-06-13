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
      const titles = { dashboard: 'Dashboard', keys: 'API Keys', usage: 'Usage & Tokens', audit: 'Audit Log', vector: 'Vector DB' };
      return titles[this.view] ?? '';
    },

    get isLoading() {
      return this.loading.dashboard || this.loading.keys || this.loading.usage || this.loading.audit || this.loading.vector;
    },

    // ── Loading flags ──────────────────────────────────────────────────────────
    loading: { dashboard: false, keys: false, usage: false, audit: false, vector: false },

    // ── Data stores ────────────────────────────────────────────────────────────
    health: { api: 'online', redis: null, vectordb: null, llm: null },
    stats:  {},
    gpu:    {},
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
      this.loginUsername      = '';
      this.view               = 'dashboard';
      this.dashboardLoaded    = false;
      this.keysLoaded         = false;
      this.usageLoaded        = false;
      this.auditLoaded        = false;
      this.auditEvents        = [];
      this.vectorCollections  = [];
      this.vectorLoaded       = false;
      this.vectorExpanded     = null;
      this.vectorFiles        = {};
      this.vectorFilesLoading = {};
      this.vectorSearch       = { query: '', appName: '', collection: '', nResults: 5, loading: false, done: false, results: [], scoreType: 'rrf', expanded: {} };
      this.stopAutoRefresh();
    },

    // ══════════════════════════════════════════════════════════════════════════
    // HTTP HELPER
    // ══════════════════════════════════════════════════════════════════════════
    async req(method, path, params = null) {
      const url = params ? path + '?' + new URLSearchParams(params) : path;
      return fetch(url, { method, headers: { Authorization: 'Basic ' + this.authHeader } });
    },

    // ══════════════════════════════════════════════════════════════════════════
    // NAVIGATION & REFRESH
    // ══════════════════════════════════════════════════════════════════════════
    async navigate(v) {
      this.view = v;
      if      (v === 'dashboard' && !this.dashboardLoaded) await this.loadDashboard();
      else if (v === 'keys'      && !this.keysLoaded)      await this.loadKeys();
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
      else if (this.view === 'usage')     await this.loadUsage();
      else if (this.view === 'vector')    { this.vectorFiles = {}; await this.loadVectorCollections(); }
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
    _adminUsage(),
    _adminAudit(),
    _adminVector(),
    _adminHelpers(),
  ].forEach(mod => Object.defineProperties(core, Object.getOwnPropertyDescriptors(mod)));

  return core;
}
