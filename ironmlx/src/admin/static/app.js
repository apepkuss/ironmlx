// ironmlx Admin Panel — Alpine.js Application

function app() {
  return {
    currentPage: 'status',
    darkMode: localStorage.getItem('theme') === 'dark' ||
      (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches),
    lang: localStorage.getItem('lang') || 'en',
    pages: [
      { id: 'status',    icon: '📊', label: 'Status' },
      { id: 'models',    icon: '🤖', label: 'Models' },
      { id: 'settings',  icon: '⚙️', label: 'Settings' },
      { id: 'logs',      icon: '📝', label: 'Logs' },
      { id: 'benchmark', icon: '⚡', label: 'Benchmark' },
      { id: 'chat',      icon: '💬', label: 'Chat' },
    ],

    // Status page data
    health: { status: 'unknown', model: '', memory: null, started_at: 0 },
    models: [],
    uptime: '—',

    // Models page data
    loadModelPath: '',
    loadModelMsg: '',
    loadModelError: false,

    init() {
      console.log('ironmlx Admin initialized');
      this.pollHealth();
      this.pollModels();
      setInterval(() => this.pollHealth(), 5000);
      setInterval(() => this.pollModels(), 10000);
      setInterval(() => this.updateUptime(), 1000);
    },

    // Utility: fetch with optional auth header
    async apiFetch(url, options = {}) {
      const apiKey = sessionStorage.getItem('ironmlx_api_key');
      if (apiKey) {
        options.headers = {
          ...options.headers,
          'Authorization': `Bearer ${apiKey}`,
        };
      }
      if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
        options.headers = {
          ...options.headers,
          'Content-Type': 'application/json',
        };
        options.body = JSON.stringify(options.body);
      }
      const resp = await fetch(url, options);
      if (resp.status === 401) {
        // TODO: show login form
        throw new Error('Unauthorized');
      }
      return resp;
    },

    // -- Polling --

    async pollHealth() {
      try {
        const resp = await this.apiFetch('/health');
        this.health = await resp.json();
      } catch (e) { console.error('Health poll failed:', e); }
    },

    async pollModels() {
      try {
        const resp = await this.apiFetch('/v1/models');
        const data = await resp.json();
        this.models = data.data || [];
      } catch (e) { console.error('Models poll failed:', e); }
    },

    updateUptime() {
      if (!this.health.started_at) { this.uptime = '—'; return; }
      const seconds = Math.floor(Date.now() / 1000 - this.health.started_at);
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = seconds % 60;
      this.uptime = `${h}h ${m}m ${s}s`;
    },

    // -- Model management --

    async loadModel() {
      this.loadModelMsg = 'Loading...';
      this.loadModelError = false;
      try {
        const resp = await this.apiFetch('/v1/models/load', {
          method: 'POST', body: { model_dir: this.loadModelPath }
        });
        if (resp.ok) {
          this.loadModelMsg = 'Model loaded successfully';
          this.loadModelPath = '';
          this.pollModels();
          this.pollHealth();
        } else {
          const data = await resp.json();
          this.loadModelMsg = data.error?.message || 'Load failed';
          this.loadModelError = true;
        }
      } catch (e) {
        this.loadModelMsg = 'Load failed: ' + e.message;
        this.loadModelError = true;
      }
    },

    async unloadModel(modelId) {
      try {
        await this.apiFetch('/v1/models/unload', { method: 'POST', body: { model: modelId } });
        this.pollModels();
        this.pollHealth();
      } catch (e) { console.error('Unload failed:', e); }
    },

    async setDefaultModel(modelId) {
      try {
        await this.apiFetch('/v1/models/default', { method: 'POST', body: { model: modelId } });
        this.pollHealth();
      } catch (e) { console.error('Set default failed:', e); }
    },
  };
}
