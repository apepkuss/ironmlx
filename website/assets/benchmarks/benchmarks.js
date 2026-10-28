// Benchmark figures for documentation pages. The build replaces
// <!-- benchmark:<figure> <id> --> with <div data-benchmark="<figure>" data-benchmark-id="<id>">
// and loads assets/benchmarks/<id>.js, which registers window.IRONMLX_BENCHMARKS[<id>].
// A data set with mode "b1" (single requests) uses the B1 metrics for the cards and the scatter axes.
// A data set with ttft_unit "s" keeps its first-token values in ms but shows them in seconds.
// The timeline grid is every 2 s for short timelines and coarser for long ones.
// A B4 data set with x_metric "prefill_tps" shows batch prefill speed in place of batch throughput.
(() => {
  const registry = window.IRONMLX_BENCHMARKS || {};
  const zh = document.documentElement.lang.startsWith("zh");
  const NS = "http://www.w3.org/2000/svg";
  const FOCUS = "A";

  const T = zh ? {
    focus: "IronMLX",
    others: "其他应用",
    ranges: (n) => `线段：${n} 轮范围`,
    lowerRight: "越靠右下越好 ↘",
    throughput: "四请求整批吞吐（tok/s）",
    ttftMedian: "首字时间中位数（ms）",
    ttftMedianSeconds: "首字时间中位数（s）",
    prefill: {
      card: ["prefill_tps", "四个请求的整批 prefill 速度", "tok/s", 0, "max"],
      throughput: "整批 prefill 速度（输入 tok/s）",
      tipThroughput: "整批 prefill",
    },
    offScale: (v) => `${v} s，超出范围（排队）`,
    seconds: "从同时发出算起的时间（秒）",
    firstToken: "首字",
    done: "回复完成",
    tipThroughput: "整批吞吐",
    tipTtft: "首字中位",
    tipRange: (n) => `${n} 轮范围`,
    tipFirst: "四个请求的首字",
    tipDone: "四个请求的完成",
    kpis: [
      ["ttft_median_ms", "首字时间中位数", "ms", 0, "min"],
      ["ttft_max_ms", "最慢请求的首字时间", "ms", 0, "min"],
      ["throughput_tps", "四个请求的整批吞吐", "tok/s", 1, "max"],
      ["memory_gib", "服务时内存峰值", "GiB", 1, "min"],
    ],
    b1: {
      kpis: [
        ["ttft_ms", "首字时间", "ms", 0, "min"],
        ["decode_tps", "解码速度", "tok/s", 1, "max"],
        ["e2e_s", "完成时间", "s", 2, "min"],
        ["memory_gib", "服务时内存峰值", "GiB", 1, "min"],
      ],
      throughput: "解码速度（tok/s）",
      ttftMedian: "首字时间（ms）",
      ttftSeconds: "首字时间（s）",
      tipThroughput: "解码速度",
      tipTtft: "首字时间",
    },
    rank: (r, n) => `${n} 个中第 ${r}`,
    firstIn: (k, n) => (k === n ? `${n} 轮都第一` : `${n} 轮中 ${k} 轮第一`),
    tied: (r, name, rounds, n) => `与 ${name} 并列第 ${r}（${n} 轮 ${rounds}）`,
    lower: (pct) => `比第 1 低 ${pct}%`,
    more: (pct) => `比第 1 多 ${pct}%`,
    slower: (pct) => `比第 1 慢 ${pct}%`,
  } : {
    focus: "IronMLX",
    others: "Other applications",
    ranges: (n) => `Lines: range over ${n} rounds`,
    lowerRight: "Better: lower right ↘",
    throughput: "4-request batch throughput (tok/s)",
    ttftMedian: "Median time to first token (ms)",
    ttftMedianSeconds: "Median time to first token (s)",
    prefill: {
      card: ["prefill_tps", "Prefill speed for four requests together", "tok/s", 0, "max"],
      throughput: "Batch prefill speed (input tok/s)",
      tipThroughput: "Batch prefill",
    },
    offScale: (v) => `${v} s, off scale (queued)`,
    seconds: "Seconds after the four requests are sent",
    firstToken: "First token",
    done: "Reply complete",
    tipThroughput: "Batch throughput",
    tipTtft: "Median first token",
    tipRange: (n) => `Range over ${n} rounds`,
    tipFirst: "First tokens (4 requests)",
    tipDone: "Completions (4 requests)",
    kpis: [
      ["ttft_median_ms", "Median time to first token", "ms", 0, "min"],
      ["ttft_max_ms", "Slowest request's first token", "ms", 0, "min"],
      ["throughput_tps", "Throughput for four requests together", "tok/s", 1, "max"],
      ["memory_gib", "Peak memory while serving", "GiB", 1, "min"],
    ],
    b1: {
      kpis: [
        ["ttft_ms", "Time to first token", "ms", 0, "min"],
        ["decode_tps", "Decode speed", "tok/s", 1, "max"],
        ["e2e_s", "Completion time", "s", 2, "min"],
        ["memory_gib", "Peak memory while serving", "GiB", 1, "min"],
      ],
      throughput: "Decode speed (tok/s)",
      ttftMedian: "Time to first token (ms)",
      ttftSeconds: "Time to first token (s)",
      tipThroughput: "Decode speed",
      tipTtft: "Time to first token",
    },
    rank: (r, n) => `#${r} of ${n}`,
    firstIn: (k, n) => (k === n ? (n === 2 ? "first in both rounds" : `first in all ${n} rounds`) : `first in ${k} of ${n} rounds`),
    tied: (r, name, rounds, n) => `Tied #${r} with ${name} (${rounds} in ${n} rounds)`,
    lower: (pct) => `${pct}% below #1`,
    more: (pct) => `${pct}% more than #1`,
    slower: (pct) => `${pct}% slower than #1`,
  };

  const fmt = (v, digits) => v.toLocaleString(zh ? "zh-CN" : "en-US", {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });

  const el = (name, attrs = {}, parent) => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (parent) parent.appendChild(node);
    return node;
  };

  const text = (parent, x, y, content, attrs = {}) => {
    const node = el("text", { x, y, ...attrs }, parent);
    node.textContent = content;
    return node;
  };

  const html = (tag, className, content) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.innerHTML = content;
    return node;
  };

  const kind = (app) => (app.key === FOCUS ? "focus" : "other");
  // Labels and metric keys for a data set: B4 (default) or B1.
  // A B4 data set with x_metric "prefill_tps" (long prompts) shows batch prefill speed in place of batch
  // throughput on the scatter's x axis and in the throughput card.
  const metrics = (data) => {
    if (data.mode === "b1") {
      return { ...T, ...T.b1, x: "decode_tps", y: "ttft_ms", ...(data.ttft_unit === "s" ? { ttftMedian: T.b1.ttftSeconds } : {}) };
    }
    const M = { ...T, x: "throughput_tps", y: "ttft_median_ms", ...(data.ttft_unit === "s" ? { ttftMedian: T.ttftMedianSeconds } : {}) };
    if (data.x_metric !== "prefill_tps") return M;
    return {
      ...M, x: "prefill_tps", throughput: T.prefill.throughput, tipThroughput: T.prefill.tipThroughput,
      kpis: T.kpis.map((k) => (k[0] === "throughput_tps" ? T.prefill.card : k)),
    };
  };
  // Display scale and unit of a card or axis: first-token values in seconds when the data set asks for it.
  const shown = (data, key, unit) => (data.ttft_unit === "s" && key.startsWith("ttft") ? [0.001, "s"] : [1, unit]);
  const tipRow = (label, value) => `<div class="tip-row"><span>${label}</span><b>${value}</b></div>`;

  const tooltip = (panel) => {
    const tip = html("div", "bench-tooltip");
    tip.hidden = true;
    panel.appendChild(tip);
    return {
      show(content, event) {
        tip.innerHTML = content;
        tip.hidden = false;
        const box = panel.getBoundingClientRect();
        let x = event.clientX - box.left + 14;
        let y = event.clientY - box.top + 14;
        if (x + tip.offsetWidth > box.width - 8) x = event.clientX - box.left - tip.offsetWidth - 14;
        if (y + tip.offsetHeight > box.height - 8) y = event.clientY - box.top - tip.offsetHeight - 14;
        tip.style.left = `${Math.max(8, x)}px`;
        tip.style.top = `${Math.max(8, y)}px`;
      },
      hide() { tip.hidden = true; },
    };
  };

  const legend = (items) => html("div", "bench-legend", items.map((item) => `<span>${item}</span>`).join(""));

  // Headline cards for IronMLX: value, rank by the median over the rounds, and how the rounds split.
  const kpis = (host, data) => {
    const M = metrics(data);
    const n = data.apps.length;
    const rounds = Object.keys(data.round_ranks[M.kpis[0][0]]).length;
    const grid = html("div", "bench-kpis");
    M.kpis.forEach(([key, label, baseUnit, defaultDigits, best]) => {
      const digits = (data.kpi_digits || {})[key] ?? defaultDigits;
      const [scale, unit] = shown(data, key, baseUnit);
      const focus = data.apps.find((a) => a.key === FOCUS);
      const sorted = [...data.apps].sort((a, b) => (best === "max" ? b[key].median - a[key].median : a[key].median - b[key].median));
      const rank = sorted.indexOf(focus) + 1;
      const perRound = Object.values(data.round_ranks[key]);
      const ahead = (x, y) => perRound.filter((order) => order.indexOf(x) < order.indexOf(y)).length;
      const neighbours = [sorted[rank - 2], sorted[rank]].filter(Boolean);
      const tie = neighbours.find((other) => ahead(FOCUS, other.key) * 2 === perRound.length);
      let note;
      let lead = false;
      if (tie) {
        const top = Math.min(rank, sorted.indexOf(tie) + 1);
        note = T.tied(top, tie.name, `${ahead(FOCUS, tie.key)}–${ahead(tie.key, FOCUS)}`, rounds);
        lead = top === 1;
      } else if (rank === 1) {
        note = `${T.rank(1, n)} · ${T.firstIn(perRound.filter((order) => order[0] === FOCUS).length, rounds)}`;
        lead = true;
      } else {
        const leader = sorted[0][key].median;
        const pct = fmt(Math.abs(focus[key].median / leader - 1) * 100, 1);
        const gap = best === "max" ? T.lower(pct) : unit === "GiB" ? T.more(pct) : T.slower(pct);
        note = `${T.rank(rank, n)} · ${gap}`;
      }
      grid.appendChild(html("div", "bench-kpi", `<span class="bench-kpi-value">${fmt(focus[key].median * scale, digits)}<small>${unit}</small></span><span class="bench-kpi-label">${label}</span><span class="bench-kpi-rank${lead ? " lead" : ""}">${note}</span>`));
    });
    host.appendChild(grid);
  };

  // Throughput vs. time to first token: one point per configuration with its range over the rounds.
  // A data set may set its own axes ([min, max, step]) and label offsets in `scatter`.
  const scatter = (host, data) => {
    const M = metrics(data);
    const axes = data.scatter || {};
    const rounds = Object.keys(Object.values(data.round_ranks)[0]).length;
    host.classList.add("bench-panel");
    host.appendChild(legend([`<i class="key-dot focus"></i>${T.focus}`, `<i class="key-dot"></i>${T.others}`, T.ranges(rounds), T.lowerRight]));
    const chart = html("div", "bench-chart");
    host.appendChild(chart);
    const W = 760, H = 430, m = { l: 64, r: 112, t: 34, b: 56 };
    const [x0, x1, xStep] = axes.x || [60, 160, 20];
    const [y0, y1, yStep] = axes.y || [100, 700, 100];
    const sx = (v) => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    const sy = (v) => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${M.throughput} / ${M.ttftMedian}` });
    for (let v = y0; v <= y1; v += yStep) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "grid" }, svg);
      text(svg, m.l - 10, sy(v) + 4, fmt(v, 0), { "text-anchor": "end" });
    }
    for (let v = x0; v <= x1; v += xStep) {
      el("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "grid" }, svg);
      text(svg, sx(v), H - m.b + 20, fmt(v, 0), { "text-anchor": "middle" });
    }
    text(svg, (m.l + W - m.r) / 2, H - 12, M.throughput, { "text-anchor": "middle", class: "axis-title" });
    text(svg, 14, m.t - 14, M.ttftMedian, { class: "axis-title" });
    const tip = tooltip(host);
    const placement = { A: [-14, -12, "end"], ...(axes.placement || {}) };
    const [ys, yUnit] = shown(data, M.y, "ms");
    const yDigits = ys === 1 ? 0 : 2;
    data.apps.filter((a) => a.key !== FOCUS).concat(data.apps.filter((a) => a.key === FOCUS)).forEach((app) => {
      const tp = app[M.x];
      const raw = app[M.y];
      const tt = { median: raw.median * ys, range: raw.range.map((v) => v * ys) };
      const g = el("g", {}, svg);
      const off = tt.median > y1;
      const cx = sx(tp.median), cy = off ? m.t : sy(tt.median);
      el("line", { x1: sx(tp.range[0]), x2: sx(tp.range[1]), y1: cy, y2: cy, class: `whisker mark-${kind(app)}`, opacity: 0.45 }, g);
      if (off) {
        el("path", { d: `M${cx - 7} ${cy + 4} L${cx} ${cy - 6} L${cx + 7} ${cy + 4} Z`, class: `mark-${kind(app)} ring` }, g);
      } else {
        el("line", { x1: cx, x2: cx, y1: sy(tt.range[0]), y2: sy(tt.range[1]), class: `whisker mark-${kind(app)}`, opacity: 0.45 }, g);
        el("circle", { cx, cy, r: app.key === FOCUS ? 7 : 5.5, class: `mark-${kind(app)} ring` }, g);
      }
      const [dx, dy, anchor] = placement[app.key] || [12, 4, "start"];
      text(g, cx + dx, cy + dy, app.name, { "text-anchor": anchor, class: app.key === FOCUS ? "label-focus" : "label-strong" });
      if (off) text(g, cx + dx, cy + 18, T.offScale(fmt(raw.median / 1000, 1)), { "text-anchor": anchor });
      const hit = el("circle", { cx, cy, r: 16, class: "hit" }, g);
      const content = `<strong>${app.name}</strong>`
        + tipRow(M.tipThroughput, `${fmt(tp.median, 1)} tok/s`)
        + tipRow(T.tipRange(rounds), `${fmt(tp.range[0], 1)}–${fmt(tp.range[1], 1)}`)
        + tipRow(M.tipTtft, `${fmt(tt.median, yDigits)} ${yUnit}`)
        + tipRow(T.tipRange(rounds), `${fmt(tt.range[0], yDigits)}–${fmt(tt.range[1], yDigits)}`);
      hit.addEventListener("mousemove", (e) => tip.show(content, e));
      hit.addEventListener("mouseleave", () => tip.hide());
    });
    chart.appendChild(svg);
    // On narrow screens the chart scrolls sideways; start at the cluster of points on the right.
    chart.scrollLeft = chart.scrollWidth;
  };

  // Request response timeline: four first tokens (ticks) and four completions (dots), medians over all batches.
  const timeline = (host, data) => {
    host.classList.add("bench-panel");
    host.appendChild(legend([`<i class="key-tick"></i>${T.firstToken}`, `<i class="key-dot focus"></i><i class="key-dot"></i>${T.done}`]));
    const chart = html("div", "bench-chart");
    host.appendChild(chart);
    const W = 760, rowH = 46, m = { l: 150, r: 24, t: 10, b: 48 };
    const H = m.t + rowH * data.apps.length + m.b;
    // Grid every 2 s for short timelines; longer ones (long prompts) take a step that keeps about ten lines.
    const last = Math.max(...data.apps.map((a) => a.order_e2e_s[3]));
    const step = [2, 5, 10, 20, 25, 50, 100].find((v) => last / v <= 12) || 100;
    const xMax = Math.ceil(last / step) * step;
    const sx = (v) => m.l + v / xMax * (W - m.l - m.r);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": T.seconds });
    for (let v = 0; v <= xMax; v += step) {
      el("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "grid" }, svg);
      text(svg, sx(v), H - m.b + 20, fmt(v, 0), { "text-anchor": "middle" });
    }
    text(svg, (m.l + W - m.r) / 2, H - 10, T.seconds, { "text-anchor": "middle", class: "axis-title" });
    const tip = tooltip(host);
    data.apps.forEach((app, i) => {
      const y = m.t + rowH * i + rowH / 2;
      const g = el("g", {}, svg);
      text(g, 0, y + 4, app.name, { class: app.key === FOCUS ? "label-focus" : "label-strong" });
      el("rect", { x: sx(0), y: y - 7, width: sx(app.order_e2e_s[3]) - sx(0), height: 14, rx: 4, class: `span-${kind(app)}` }, g);
      app.order_ttft_ms.forEach((ms) => {
        const x = sx(ms / 1000);
        el("line", { x1: x, x2: x, y1: y - 10, y2: y + 10, class: "tick-mark" }, g);
      });
      app.order_e2e_s.forEach((s) => el("circle", { cx: sx(s), cy: y, r: 5, class: `mark-${kind(app)} ring` }, g));
      const hit = el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, class: "hit" }, g);
      const content = `<strong>${app.name}</strong>`
        + tipRow(T.tipFirst, app.order_ttft_ms.map((v) => (v < 1000 ? `${fmt(v, 0)} ms` : `${fmt(v / 1000, 1)} s`)).join(" · "))
        + tipRow(T.tipDone, app.order_e2e_s.map((v) => `${fmt(v, 1)} s`).join(" · "));
      hit.addEventListener("mousemove", (e) => tip.show(content, e));
      hit.addEventListener("mouseleave", () => tip.hide());
    });
    chart.appendChild(svg);
  };

  const figures = { kpis, scatter, timeline };
  document.querySelectorAll("[data-benchmark]").forEach((host) => {
    const data = registry[host.dataset.benchmarkId];
    const render = figures[host.dataset.benchmark];
    if (data && render) render(host, data);
  });

  // Results tables: in a table whose header has an IronMLX column, tint that column and shade each body cell
  // whose value is bold (the best value of its row); tables without an IronMLX column are left as they are.
  document.querySelectorAll(".doc-content table").forEach((table) => {
    const head = table.tHead ? table.tHead.rows[0] : table.rows[0];
    if (!head) return;
    const focus = [...head.cells].findIndex((cell) => cell.textContent.trim().startsWith("IronMLX"));
    if (focus < 0) return;
    [...table.rows].forEach((row) => {
      const body = row.parentElement.tagName === "TBODY";
      [...row.cells].forEach((cell, i) => {
        if (i === focus) cell.classList.add("bench-focus-col");
        const first = cell.firstElementChild;
        if (body && i > 0 && first && first.tagName === "STRONG" && cell.textContent.trim().startsWith(first.textContent.trim())) {
          cell.classList.add("bench-best");
        }
      });
    });
  });
})();
