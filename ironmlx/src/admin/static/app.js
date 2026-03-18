// ironmlx Admin Panel — Alpine.js Application

function app() {
  return {
    currentPage: 'status',
    themeMode: localStorage.getItem('themeMode') || 'system',
    darkMode: false,
    lang: localStorage.getItem('lang') || 'en',
    pages: [
      { id: 'status',    icon: '📊' },
      { id: 'models',    icon: '🤖' },
      { id: 'settings',  icon: '⚙️' },
      { id: 'logs',      icon: '📝' },
      { id: 'benchmark', icon: '⚡' },
      { id: 'chat',      icon: '💬' },
    ],

    // Status page data
    health: { status: 'unknown', model: '', memory: null, started_at: 0 },
    models: [],
    uptime: '—',

    // Models page data
    loadModelPath: '',
    loadModelMsg: '',
    loadModelError: false,
    // HuggingFace search/download
    hfSearchQuery: '',
    hfSearchResults: [],
    hfSearching: false,
    hfDownloads: [],
    hfDownloadPolling: null,

    // Settings page data
    settings: {
      host: '', port: 0, memory_limit_gb: 0, cache_max_size_gb: 0,
      max_num_seqs: 256, temperature: 1.0, top_p: 1.0,
      api_key: '', api_key_set: false, log_level: 'info',
      hf_endpoint: 'https://huggingface.co',
      chat_template_override: '',
      model_aliases: {},
      log_buffer_size: 100,
    },
    settingsMsg: '',
    settingsError: false,
    newAliasName: '',
    newAliasTarget: '',

    // Logs page data
    logs: [],
    logLevel: 'all',

    // Benchmark page data
    benchModel: '',
    benchPrompt: 'Explain the theory of relativity in simple terms.',
    benchTokens: 50,
    benchRunning: false,
    benchResult: null,
    benchHistory: [],

    // Chat page data
    chatMessages: [],
    chatInput: '',
    chatModel: '',
    chatStreaming: false,
    chatStreamContent: '',

    $t(key) {
      const dict = translations[this.lang] || translations.en;
      return dict[key] || translations.en[key] || key;
    },

    applyTheme() {
      if (this.themeMode === 'system') {
        this.darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
      } else {
        this.darkMode = this.themeMode === 'dark';
      }
    },
    setThemeMode(mode) {
      this.themeMode = mode;
      localStorage.setItem('themeMode', mode);
      this.applyTheme();
    },
    init() {
      console.log('ironmlx Admin initialized');
      this.applyTheme();
      // Listen for system theme changes
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (this.themeMode === 'system') this.applyTheme();
      });
      this.pollHealth();
      this.pollModels();
      setInterval(() => this.pollHealth(), 5000);
      setInterval(() => this.pollModels(), 10000);
      setInterval(() => this.updateUptime(), 1000);
      // Set default chat/bench model after first health poll
      setTimeout(() => {
        if (this.health.model) {
          this.chatModel = this.health.model;
          this.benchModel = this.health.model;
        }
      }, 2000);
      // Poll logs every 3 seconds
      this.pollLogs();
      setInterval(() => this.pollLogs(), 3000);
      // Fetch settings/history when switching pages
      this.$watch('currentPage', (page) => {
        if (page === 'settings') this.fetchSettings();
        if (page === 'benchmark') this.fetchBenchHistory();
      });
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
          this.loadModelMsg = this.$t('models.load_success');
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

    // -- Settings --

    async fetchSettings() {
      try {
        const resp = await this.apiFetch('/admin/api/settings');
        const data = await resp.json();
        this.settings = {
          ...data,
          api_key: '',
          chat_template_override: data.chat_template_override || '',
          model_aliases: data.model_aliases || {},
        };
      } catch (e) { console.error('Settings fetch failed:', e); }
    },

    async saveSettings() {
      this.settingsMsg = '';
      this.settingsError = false;
      try {
        const body = {
          memory_limit_gb: this.settings.memory_limit_gb,
          cache_max_size_gb: this.settings.cache_max_size_gb,
          max_num_seqs: this.settings.max_num_seqs,
          temperature: this.settings.temperature,
          top_p: this.settings.top_p,
          log_level: this.settings.log_level,
          hf_endpoint: this.settings.hf_endpoint,
          chat_template_override: this.settings.chat_template_override || '',
          model_aliases: this.settings.model_aliases,
          log_buffer_size: this.settings.log_buffer_size,
        };
        // Only send api_key if the user typed something
        if (this.settings.api_key !== '') {
          body.api_key = this.settings.api_key;
        }
        const resp = await this.apiFetch('/admin/api/settings', {
          method: 'POST', body: body,
        });
        if (resp.ok) {
          this.settingsMsg = this.$t('settings.saved');
          // If api_key was changed, store it in sessionStorage
          if (this.settings.api_key) {
            sessionStorage.setItem('ironmlx_api_key', this.settings.api_key);
          }
          this.fetchSettings();
        } else {
          this.settingsMsg = 'Save failed';
          this.settingsError = true;
        }
      } catch (e) {
        this.settingsMsg = 'Save failed: ' + e.message;
        this.settingsError = true;
      }
    },

    addAlias() {
      const name = this.newAliasName.trim();
      const target = this.newAliasTarget.trim();
      if (!name || !target) return;
      this.settings.model_aliases = { ...this.settings.model_aliases, [name]: target };
      this.newAliasName = '';
      this.newAliasTarget = '';
    },

    removeAlias(alias) {
      const copy = { ...this.settings.model_aliases };
      delete copy[alias];
      this.settings.model_aliases = copy;
    },

    // -- Logs --

    get filteredLogs() {
      if (this.logLevel === 'all') return this.logs;
      return this.logs.filter(l => l.level === this.logLevel);
    },

    async pollLogs() {
      try {
        const resp = await this.apiFetch('/admin/api/logs');
        this.logs = await resp.json();
      } catch(e) {}
    },

    clearLogs() { this.logs = []; },

    // -- Benchmark --

    async runBenchmark() {
      this.benchRunning = true;
      this.benchResult = null;
      try {
        const resp = await this.apiFetch('/admin/api/benchmark', {
          method: 'POST',
          body: {
            model: this.benchModel || undefined,
            prompt: this.benchPrompt,
            max_tokens: this.benchTokens,
          },
        });
        if (resp.ok) {
          this.benchResult = await resp.json();
          this.fetchBenchHistory();
        }
      } catch(e) { console.error('Benchmark failed:', e); }
      this.benchRunning = false;
    },

    async fetchBenchHistory() {
      try {
        const resp = await this.apiFetch('/admin/api/benchmark/history');
        if (resp.ok) {
          this.benchHistory = await resp.json();
        }
      } catch(e) { console.error('Bench history fetch failed:', e); }
    },

    async clearBenchHistory() {
      try {
        await this.apiFetch('/admin/api/benchmark/history', { method: 'DELETE' });
        this.benchHistory = [];
      } catch(e) { console.error('Clear bench history failed:', e); }
    },

    // -- HuggingFace Search & Download --

    async hfSearch() {
      if (!this.hfSearchQuery.trim()) return;
      this.hfSearching = true;
      this.hfSearchResults = [];
      try {
        const resp = await this.apiFetch('/admin/api/models/search', {
          method: 'POST', body: { query: this.hfSearchQuery },
        });
        if (resp.ok) {
          this.hfSearchResults = await resp.json();
        }
      } catch (e) { console.error('HF search failed:', e); }
      this.hfSearching = false;
    },

    async hfDownload(repoId) {
      try {
        await this.apiFetch('/admin/api/models/download', {
          method: 'POST', body: { repo_id: repoId },
        });
        this.startDownloadPolling();
      } catch (e) { console.error('HF download failed:', e); }
    },

    startDownloadPolling() {
      if (this.hfDownloadPolling) return;
      this.pollDownloads();
      this.hfDownloadPolling = setInterval(() => this.pollDownloads(), 2000);
    },

    stopDownloadPolling() {
      if (this.hfDownloadPolling) {
        clearInterval(this.hfDownloadPolling);
        this.hfDownloadPolling = null;
      }
    },

    async pollDownloads() {
      try {
        const resp = await this.apiFetch('/admin/api/models/downloads');
        this.hfDownloads = await resp.json();
        // Stop polling if no active downloads
        const active = this.hfDownloads.some(d => d.status === 'downloading');
        if (!active && this.hfDownloads.length > 0) {
          this.stopDownloadPolling();
        }
      } catch (e) { console.error('Download poll failed:', e); }
    },

    hfDownloadStatusColor(status) {
      if (status === 'completed') return 'text-green-400';
      if (status === 'failed') return 'text-red-400';
      return 'text-blue-400';
    },

    isDownloading(repoId) {
      return this.hfDownloads.some(d => d.repo_id === repoId && d.status === 'downloading');
    },

    // -- Chat --

    clearChat() {
      this.chatMessages = [];
      this.chatStreamContent = '';
    },

    escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    },

    renderMarkdown(text) {
      if (!text) return '';
      try {
        const html = marked.parse(text);
        return html;
      } catch (e) {
        return this.escapeHtml(text);
      }
    },

    renderMathInChat() {
      if (typeof renderMathInElement !== 'function') return;
      const container = this.$refs.chatMessages;
      if (!container) return;
      renderMathInElement(container, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
        ],
        throwOnError: false,
      });
    },

    async sendChat() {
      const input = this.chatInput.trim();
      console.log('sendChat called, input:', JSON.stringify(input), 'streaming:', this.chatStreaming, 'model:', this.chatModel);
      if (!input || this.chatStreaming) return;

      // Add user message
      this.chatMessages.push({ role: 'user', content: input });
      this.chatInput = '';
      this.chatStreaming = true;
      this.chatStreamContent = '';

      // Scroll to bottom
      this.$nextTick(() => {
        if (this.$refs.chatMessages) {
          this.$refs.chatMessages.scrollTop = this.$refs.chatMessages.scrollHeight;
        }
      });

      try {
        // Build messages array — only include messages with content
        const messages = this.chatMessages
          .filter(m => m.content && m.content.trim())
          .map(m => ({ role: m.role, content: m.content }));

        const body = JSON.stringify({
          model: this.chatModel || undefined,
          messages: messages,
          stream: true,
          max_tokens: 2048,
          temperature: 0.7,
        });

        console.log('Sending fetch...');
        const resp = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: body,
        });
        console.log('Fetch response:', resp.status, resp.headers.get('content-type'));

        if (!resp.ok) {
          this.chatStreamContent = 'Error: ' + resp.status + ' ' + resp.statusText;
          this.chatStreaming = false;
          return;
        }

        // Read SSE stream
        const reader = resp.body.getReader();
        console.log('Reading stream...');
        const decoder = new TextDecoder();
        let buffer = '';
        let streamDone = false;

        while (!streamDone) {
          const { done, value } = await reader.read();
          console.log('Stream read:', done, value ? value.length + ' bytes' : 'null');
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // Keep incomplete line

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data: ')) continue;
            const data = trimmed.slice(6);
            if (data === '[DONE]') { streamDone = true; break; }

            try {
              const chunk = JSON.parse(data);
              const delta = chunk.choices?.[0]?.delta;
              if (delta?.content) {
                this.chatStreamContent += delta.content;
              }
              if (delta?.reasoning_content) {
                this.chatStreamContent += delta.reasoning_content;
              }
            } catch (e) { /* ignore parse errors */ }
          }

          // Scroll during streaming
          this.$nextTick(() => {
            if (this.$refs.chatMessages) {
              this.$refs.chatMessages.scrollTop = this.$refs.chatMessages.scrollHeight;
            }
          });
        }
      } catch (e) {
        this.chatStreamContent = 'Error: ' + e.message;
      }

      // Move stream content to messages
      if (this.chatStreamContent) {
        this.chatMessages.push({ role: 'assistant', content: this.chatStreamContent });
      }
      this.chatStreaming = false;
      this.chatStreamContent = '';

      // Highlight code blocks and render math
      this.$nextTick(() => {
        document.querySelectorAll('pre code').forEach(block => {
          hljs.highlightElement(block);
        });
        this.renderMathInChat();
      });
    },
  };
}
