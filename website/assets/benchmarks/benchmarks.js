// Benchmark figures for documentation pages. The build replaces
// <!-- benchmark:<figure> <id> --> with <div data-benchmark="<figure>" data-benchmark-id="<id>">
// and loads assets/benchmarks/<id>.js, which registers window.IRONMLX_BENCHMARKS[<id>].
(() => {
  const registry = window.IRONMLX_BENCHMARKS || {};
  const zh = document.documentElement.lang.startsWith("zh");
  const NS = "http://www.w3.org/2000/svg";
  const FOCUS = "A";

  const T = zh ? {
    focus: "IronMLX",
    others: "其他应用",
    ranges: "线段：四轮范围",
    lowerRight: "越靠右下越好 ↘",
    throughput: "四请求整批吞吐（tok/s）",
    ttftMedian: "首字时间中位数（ms）",
    offScale: (v) => `${v} s，超出范围（排队）`,
    seconds: "从同时发出算起的时间（秒）",
    firstToken: "首字",
    done: "回复完成",
    tipThroughput: "整批吞吐",
    tipTtft: "首字中位",
    tipRange: "四轮范围",
    tipFirst: "四个请求的首字",
    tipDone: "四个请求的完成",
    kpis: [
      ["ttft_median_ms", "首字时间中位数", "ms", 0, "min"],
      ["ttft_max_ms", "最慢请求的首字时间", "ms", 0, "min"],
      ["throughput_tps", "四个请求的整批吞吐", "tok/s", 0, "max"],
      ["memory_gib", "服务时内存峰值", "GiB", 1, "min"],
    ],
    rank: (r, n) => `${n} 个中第 ${r}`,
    firstIn: (k, n) => (k === n ? `${n} 轮都第一` : `${n} 轮中 ${k} 轮第一`),
    tied: (r, name, rounds) => `与 ${name} 并列第 ${r}（四轮 ${rounds}）`,
    lower: (pct) => `比第 1 低 ${pct}%`,
    more: (pct) => `比第 1 多 ${pct}%`,
    slower: (pct) => `比第 1 慢 ${pct}%`,
  } : {
    focus: "IronMLX",
    others: "Other applications",
    ranges: "Lines: range over four rounds",
    lowerRight: "Better: lower right ↘",
    throughput: "4-request batch throughput (tok/s)",
    ttftMedian: "Median time to first token (ms)",
    offScale: (v) => `${v} s, off scale (queued)`,
    seconds: "Seconds after the four requests are sent",
    firstToken: "First token",
    done: "Reply complete",
    tipThroughput: "Batch throughput",
    tipTtft: "Median first token",
    tipRange: "Range over 4 rounds",
    tipFirst: "First tokens (4 requests)",
    tipDone: "Completions (4 requests)",
    kpis: [
      ["ttft_median_ms", "Median time to first token", "ms", 0, "min"],
      ["ttft_max_ms", "Slowest request's first token", "ms", 0, "min"],
      ["throughput_tps", "Throughput for four requests together", "tok/s", 0, "max"],
      ["memory_gib", "Peak memory while serving", "GiB", 1, "min"],
    ],
    rank: (r, n) => `#${r} of ${n}`,
    firstIn: (k, n) => (k === n ? `first in all ${n} rounds` : `first in ${k} of ${n} rounds`),
    tied: (r, name, rounds) => `Tied #${r} with ${name} (${rounds} in rounds)`,
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

  // Headline cards for IronMLX: value, rank by the four-round median, and how the rounds split.
  const kpis = (host, data) => {
    const n = data.apps.length;
    const rounds = Object.keys(data.round_ranks.throughput_tps).length;
    const grid = html("div", "bench-kpis");
    T.kpis.forEach(([key, label, unit, digits, best]) => {
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
        note = T.tied(top, tie.name, `${ahead(FOCUS, tie.key)}–${ahead(tie.key, FOCUS)}`);
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
      grid.appendChild(html("div", "bench-kpi", `<span class="bench-kpi-value">${fmt(focus[key].median, digits)}<small>${unit}</small></span><span class="bench-kpi-label">${label}</span><span class="bench-kpi-rank${lead ? " lead" : ""}">${note}</span>`));
    });
    host.appendChild(grid);
  };

  // Throughput vs. time to first token: one point per configuration with four-round ranges.
  const scatter = (host, data) => {
    host.classList.add("bench-panel");
    host.appendChild(legend([`<i class="key-dot focus"></i>${T.focus}`, `<i class="key-dot"></i>${T.others}`, T.ranges, T.lowerRight]));
    const chart = html("div", "bench-chart");
    host.appendChild(chart);
    const W = 760, H = 430, m = { l: 64, r: 112, t: 34, b: 56 };
    const x0 = 60, x1 = 160, y0 = 100, y1 = 700;
    const sx = (v) => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    const sy = (v) => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${T.throughput} / ${T.ttftMedian}` });
    for (let v = y0; v <= y1; v += 100) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "grid" }, svg);
      text(svg, m.l - 10, sy(v) + 4, fmt(v, 0), { "text-anchor": "end" });
    }
    for (let v = x0; v <= x1; v += 20) {
      el("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "grid" }, svg);
      text(svg, sx(v), H - m.b + 20, fmt(v, 0), { "text-anchor": "middle" });
    }
    text(svg, (m.l + W - m.r) / 2, H - 12, T.throughput, { "text-anchor": "middle", class: "axis-title" });
    text(svg, 14, m.t - 14, T.ttftMedian, { class: "axis-title" });
    const tip = tooltip(host);
    const placement = { A: [-14, -12, "end"] };
    data.apps.filter((a) => a.key !== FOCUS).concat(data.apps.filter((a) => a.key === FOCUS)).forEach((app) => {
      const tp = app.throughput_tps, tt = app.ttft_median_ms;
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
      if (off) text(g, cx + dx, cy + 18, T.offScale(fmt(tt.median / 1000, 1)), { "text-anchor": anchor });
      const hit = el("circle", { cx, cy, r: 16, class: "hit" }, g);
      const content = `<strong>${app.name}</strong>`
        + tipRow(T.tipThroughput, `${fmt(tp.median, 1)} tok/s`)
        + tipRow(T.tipRange, `${fmt(tp.range[0], 1)}–${fmt(tp.range[1], 1)}`)
        + tipRow(T.tipTtft, `${fmt(tt.median, 0)} ms`)
        + tipRow(T.tipRange, `${fmt(tt.range[0], 0)}–${fmt(tt.range[1], 0)}`);
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
    const xMax = Math.ceil(Math.max(...data.apps.map((a) => a.order_e2e_s[3])) / 2) * 2;
    const sx = (v) => m.l + v / xMax * (W - m.l - m.r);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": T.seconds });
    for (let v = 0; v <= xMax; v += 2) {
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
})();
