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

    // Chat page data
    chatMessages: [],
    chatInput: '',
    chatModel: '',
    chatStreaming: false,
    chatStreamContent: '',

    init() {
      console.log('ironmlx Admin initialized');
      this.pollHealth();
      this.pollModels();
      setInterval(() => this.pollHealth(), 5000);
      setInterval(() => this.pollModels(), 10000);
      setInterval(() => this.updateUptime(), 1000);
      // Set default chat model after first health poll
      setTimeout(() => { if (this.health.model) this.chatModel = this.health.model; }, 2000);
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

    async sendChat() {
      const input = this.chatInput.trim();
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
        // Build messages array for API
        const messages = this.chatMessages.map(m => ({
          role: m.role,
          content: m.content,
        }));

        const resp = await this.apiFetch('/v1/chat/completions', {
          method: 'POST',
          body: {
            model: this.chatModel || undefined,
            messages: messages,
            stream: true,
            max_tokens: 2048,
            temperature: 0.7,
          },
        });

        // Read SSE stream
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // Keep incomplete line

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6);
            if (data === '[DONE]') break;

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

      // Highlight code blocks
      this.$nextTick(() => {
        document.querySelectorAll('pre code').forEach(block => {
          hljs.highlightElement(block);
        });
      });
    },
  };
}
