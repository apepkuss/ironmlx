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

    init() {
      console.log('ironmlx Admin initialized');
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
  };
}
