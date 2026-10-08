(() => {
  const chinese = document.body.dataset.docLanguage === "zh";
  const label = (en, zh) => (chinese ? zh : en);
  const dialog = document.querySelector(".docs-search-dialog");
  const input = document.querySelector("#docs-search-input");
  const results = document.querySelector(".search-results");
  const status = document.querySelector(".search-status");
  const navigation = document.querySelector(".docs-navigation-dialog");
  const searchURL = new URL(document.body.dataset.searchIndex, location.href);
  let index;
  let selected = 0;
  let loading;

  const normalize = (value) => value.toLocaleLowerCase().normalize("NFKC");
  const plain = (value) => value.replace(/[#`*|>]/g, "").replace(/\s+/g, " ").trim();

  const highlight = (element, value, terms) => {
    // Construct nodes rather than interpreting search terms or source text as HTML.
    if (!terms.length) {
      element.textContent = value;
      return;
    }
    const escaped = terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const pattern = new RegExp(`(${escaped.join("|")})`, "giu");
    let offset = 0;
    for (const match of value.matchAll(pattern)) {
      element.append(document.createTextNode(value.slice(offset, match.index)));
      const mark = document.createElement("mark");
      mark.textContent = match[0];
      element.append(mark);
      offset = match.index + match[0].length;
    }
    element.append(document.createTextNode(value.slice(offset)));
  };

  const excerpt = (value, terms) => {
    const clean = plain(value);
    const lower = normalize(clean);
    const positions = terms.map((term) => lower.indexOf(term)).filter((position) => position >= 0);
    const first = positions.length ? Math.min(...positions) : 0;
    const start = Math.max(0, first - 32);
    return `${start ? "…" : ""}${clean.slice(start, start + 145)}${clean.length > start + 145 ? "…" : ""}`;
  };

  const selectResult = (position, scroll = false) => {
    const links = [...results.querySelectorAll("a")];
    if (!links.length) return;
    selected = (position + links.length) % links.length;
    links.forEach((link, i) => {
      link.dataset.selected = String(i === selected);
      if (i === selected && scroll) link.scrollIntoView({ block: "nearest" });
    });
  };

  const renderResults = () => {
    if (!index) return;
    const query = normalize(input.value.trim());
    const terms = [...new Set(query.split(/\s+/).filter(Boolean))];
    const matches = [];
    for (const doc of index) {
      const title = normalize(doc.title);
      const full = normalize(`${doc.title} ${doc.group} ${doc.text}`);
      if (!terms.every((term) => full.includes(term))) continue;
      const titleMatch = terms.every((term) => title.includes(term));
      let section;
      if (terms.length && !titleMatch) {
        section = doc.sections
          .filter((item) => terms.every((term) => normalize(`${doc.title} ${item.title} ${item.text}`).includes(term)))
          .sort((a, b) => terms.filter((term) => normalize(b.title).includes(term)).length - terms.filter((term) => normalize(a.title).includes(term)).length)[0];
      }
      const score = terms.reduce((total, term) => total + (title.includes(term) ? 20 : 0) + (normalize(section?.title || "").includes(term) ? 10 : 0), 0);
      matches.push({ doc, section, score });
    }
    matches.sort((a, b) => b.score - a.score);
    results.replaceChildren();
    selected = 0;
    const visible = matches.slice(0, terms.length ? 20 : 6);
    status.textContent = terms.length
      ? (matches.length ? label(`${matches.length} matching documents`, `找到 ${matches.length} 篇相关文档`) : label("No matches. Try a different keyword.", "没有找到相关内容，请尝试其他关键词。"))
      : label("Browse documentation", "浏览文档");
    for (const { doc, section } of visible) {
      const link = document.createElement("a");
      const href = new URL(doc.href, searchURL);
      if (section) href.hash = section.anchor;
      link.href = href.href;
      const group = document.createElement("span");
      group.className = "search-result-group";
      group.textContent = doc.group;
      const title = document.createElement("strong");
      highlight(title, section ? `${doc.title} › ${section.title}` : doc.title, terms);
      const description = document.createElement("span");
      description.className = "search-excerpt";
      highlight(description, terms.length ? excerpt(section?.text || doc.text, terms) : doc.description, terms);
      link.append(group, title, description);
      link.addEventListener("pointermove", () => selectResult([...results.querySelectorAll("a")].indexOf(link)));
      link.addEventListener("focus", () => selectResult([...results.querySelectorAll("a")].indexOf(link)));
      link.addEventListener("click", () => dialog.close());
      results.append(link);
    }
    selectResult(0);
  };

  const loadIndex = async () => {
    if (index) return;
    if (loading) return loading;
    status.textContent = label("Loading documentation…", "正在加载文档…");
    results.replaceChildren();
    loading = (async () => {
      try {
        const response = await fetch(searchURL);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!Array.isArray(data)) throw new Error("Invalid search index");
        index = data;
        renderResults();
      } catch (_) {
        status.textContent = label("Search could not load. Please try again.", "搜索加载失败，请重试。");
        const retry = document.createElement("button");
        retry.type = "button";
        retry.className = "search-retry";
        retry.textContent = label("Retry", "重新加载");
        retry.addEventListener("click", loadIndex);
        results.replaceChildren(retry);
      } finally {
        loading = null;
      }
    })();
    return loading;
  };

  const openSearch = () => {
    if (navigation.open) navigation.close();
    if (!dialog.open) dialog.showModal();
    input.focus();
    loadIndex();
    renderResults();
  };

  document.querySelector("[data-open-search]").addEventListener("click", openSearch);
  document.querySelector("[data-close-search]").addEventListener("click", () => dialog.close());
  input.addEventListener("input", renderResults);
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      dialog.close();
      return;
    }
    if (!["ArrowDown", "ArrowUp", "Enter"].includes(event.key) || !results.querySelector("a")) return;
    if (event.key === "Enter" && event.target === input) {
      event.preventDefault();
      results.querySelectorAll("a")[selected]?.click();
    } else if (event.key !== "Enter") {
      event.preventDefault();
      selectResult(selected + (event.key === "ArrowDown" ? 1 : -1), true);
      if (event.target !== input) results.querySelectorAll("a")[selected]?.focus();
    }
  });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      if (dialog.open) dialog.close();
      else openSearch();
    }
  });
  if (!/Mac|iPhone|iPad/.test(navigator.platform)) {
    document.querySelector("[data-search-shortcut]").textContent = "Ctrl K";
  }

  document.querySelector("[data-open-navigation]").addEventListener("click", () => navigation.showModal());
  document.querySelector("[data-close-navigation]").addEventListener("click", () => navigation.close());
  for (const modal of [dialog, navigation]) {
    modal.addEventListener("click", (event) => {
      const box = modal.getBoundingClientRect();
      if (event.target === modal && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) modal.close();
    });
    modal.addEventListener("close", () => {
      document.body.classList.toggle("docs-modal-open", dialog.open || navigation.open);
    });
  }
  const modalObserver = new MutationObserver(() => {
    document.body.classList.toggle("docs-modal-open", dialog.open || navigation.open);
  });
  modalObserver.observe(dialog, { attributes: true, attributeFilter: ["open"] });
  modalObserver.observe(navigation, { attributes: true, attributeFilter: ["open"] });
  const desktop = matchMedia("(min-width: 681px)");
  desktop.addEventListener("change", () => {
    if (desktop.matches && navigation.open) navigation.close();
  });

  // A small local tokenizer covers the shell, JSON, and code examples in these docs.
  // Tokens are text nodes; no source content is ever executed or parsed as HTML.
  for (const block of document.querySelectorAll(".code-block")) {
    const code = block.querySelector("code");
    const button = block.querySelector("[data-copy-code]");
    const source = code.textContent;
    const language = code.className.replace("language-", "");
    if (["bash", "sh", "shell", "json", "python", "rust", "javascript", "js", "toml"].includes(language)) {
      const tokenPattern = /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#[^\n]*|\b(?:true|false|null|None|True|False|def|return|import|from|as|if|else|for|in|fn|pub|let|mut|use|struct|impl|const|async|await)\b|\b\d+(?:\.\d+)?\b|\$[A-Za-z_][\w]*|(?:^|\s)--?[A-Za-z][\w-]*/gm;
      code.replaceChildren();
      let offset = 0;
      for (const match of source.matchAll(tokenPattern)) {
        code.append(document.createTextNode(source.slice(offset, match.index)));
        const token = document.createElement("span");
        const text = match[0];
        let kind = "keyword";
        if (/^["']/.test(text)) kind = language === "json" && /^\s*:/.test(source.slice(match.index + text.length)) ? "property" : "string";
        else if (text.startsWith("#")) kind = "comment";
        else if (/^\d/.test(text)) kind = "number";
        else if (/^\s*-/.test(text) || text.startsWith("$")) kind = "flag";
        token.className = `token-${kind}`;
        token.textContent = text;
        code.append(token);
        offset = match.index + text.length;
      }
      code.append(document.createTextNode(source.slice(offset)));
    }
    let resetTimer;
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(source);
        button.textContent = label("Copied", "已复制");
        document.querySelector("[data-copy-status]").textContent = label("Code copied to clipboard", "代码已复制到剪贴板");
      } catch (_) {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(code);
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = label("Press Ctrl/⌘ C", "请按 Ctrl/⌘ C");
        document.querySelector("[data-copy-status]").textContent = label("Code selected. Use your copy shortcut.", "代码已选中，请使用复制快捷键。");
      }
      clearTimeout(resetTimer);
      resetTimer = setTimeout(() => { button.textContent = label("Copy", "复制"); }, 2200);
    });
  }

  const headings = [...document.querySelectorAll(".doc-content h2[id], .doc-content h3[id]")];
  const outlineLinks = [...document.querySelectorAll(".docs-outline a")];
  if (headings.length && outlineLinks.length) {
    let pending = false;
    const syncOutline = () => {
      pending = false;
      let active = headings[0];
      for (const heading of headings) {
        if (heading.getBoundingClientRect().top <= 140) active = heading;
      }
      outlineLinks.forEach((link) => {
        if (decodeURIComponent(link.hash.slice(1)) === active.id) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    };
    document.addEventListener("scroll", () => {
      if (!pending) {
        pending = true;
        requestAnimationFrame(syncOutline);
      }
    }, { passive: true });
    window.addEventListener("resize", syncOutline);
    syncOutline();
  }
  const sidebar = document.querySelector(".docs-sidebar");
  const current = sidebar.querySelector('[aria-current="page"]');
  if (current && current.offsetTop > sidebar.clientHeight - 50) sidebar.scrollTop = current.offsetTop - sidebar.clientHeight / 2;
})();
