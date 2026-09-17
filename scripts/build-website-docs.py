#!/usr/bin/env python3
"""Render the maintained user-facing Markdown subset into the static website."""
from html import escape
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "website" / "docs"
ZH_OUT = ROOT / "website" / "zh-Hans" / "docs"
PAGES = (
    ("user-guide.md", "User guide", "用户指南"),
    ("supported-models.md", "Supported models", "支持的模型"),
    ("api.md", "HTTP API quick start", "HTTP API 快速开始"),
    ("dflash2-server-api.md", "DFlash2 server API", "DFlash2 服务端 API"),
    ("api-reference.md", "API reference", "API 参考"),
    ("api-compatibility-matrix.md", "API compatibility matrix", "API 兼容矩阵"),
    ("automatic-updates.md", "Automatic updates", "自动更新"),
    ("building-from-source.md", "Building from source", "从源码构建"),
    ("diagnostic-bundle.md", "Diagnostic export", "诊断信息导出"),
    ("engine-pool.md", "Engine pool", "Engine Pool"),
    ("known-issues.md", "Known issues", "已知问题"),
    ("model-license-boundary.md", "Model rights boundary", "模型权利边界"),
    ("mtp-server-api.md", "MTP server API", "MTP 服务端 API"),
    ("scheduler-profile-v5.md", "Scheduler profile", "Scheduler 配置"),
    ("security-boundary.md", "Security boundary", "安全边界"),
    ("stable-release-pipeline.md", "Release pipeline", "发布流水线"),
    ("storage-and-uninstall.md", "Data locations and uninstall", "数据位置与卸载"),
    ("versioning-and-releases.md", "Versioning and releases", "版本与发布"),
    ("hermes-agent.md", "Hermes Agent", "Hermes Agent"),
    ("oh-my-pi.md", "oh-my-pi", "oh-my-pi"),
    ("troubleshooting.md", "Troubleshooting", "故障排查"),
    ("privacy.md", "Privacy", "隐私说明"),
    ("release-notes/0.1.0.md", "0.1.0 release notes", "0.1.0 发布说明"),
)


def inline(text, source, target):
    output_path = target
    text = escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    def image(match):
        alt, original = match.group(1), match.group(2)
        if original.startswith("images/") or original.startswith("../images/"):
            asset = ROOT / "website" / "assets" / Path(original).name
            original = Path(__import__("os").path.relpath(asset, output_path.parent)).as_posix()
        return f'<img src="{escape(original, quote=True)}" alt="{alt}">'
    text = re.sub(r"!\[([^]]*)\]\(([^)]+)\)", image, text)
    def link(match):
        label, target = match.group(1), match.group(2)
        if target.endswith(".md"):
            source_target = (source.parent / target).resolve()
            if source_target.is_relative_to(ROOT / "docs"):
                if source_target.is_relative_to(ROOT / "docs" / "zh-CN"):
                    generated = ROOT / "website" / "zh-Hans" / "docs" / source_target.relative_to(ROOT / "docs" / "zh-CN")
                else:
                    generated = ROOT / "website" / "docs" / source_target.relative_to(ROOT / "docs")
                if source_target.name in {"contributing.md", "support.md"}:
                    target = f"https://github.com/apepkuss/ironmlx/blob/main/{source_target.relative_to(ROOT).as_posix()}"
                else:
                    target = Path(__import__("os").path.relpath(generated.with_suffix(".html"), output_path.parent)).as_posix()
            else:
                target = f"https://github.com/apepkuss/ironmlx/blob/main/{source_target.relative_to(ROOT).as_posix()}" if source_target.is_relative_to(ROOT) else Path(target).with_suffix(".html").as_posix()
        return f'<a href="{escape(target, quote=True)}">{label}</a>'
    return re.sub(r"\[([^]]+)\]\(([^)]+)\)", link, text)


def render(markdown, source, target):
    raw_lines = markdown.splitlines()
    lines = []
    for raw in raw_lines:
        if raw[:1].isspace() and lines and re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", lines[-1]):
            lines[-1] += " " + raw.strip()
        else:
            lines.append(raw)
    lines, html, toc = lines, [], []
    in_code = False
    list_tag = None
    table = False
    table_header_written = False
    paragraph = []
    def flush_paragraph():
        if paragraph:
            html.append(f"<p>{inline(' '.join(paragraph), source, target)}</p>")
            paragraph.clear()
    def close_list():
        nonlocal list_tag
        if list_tag:
            html.append(f"</{list_tag}>")
            list_tag = None
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph(); close_list()
            if in_code:
                html.append("</code></pre>")
            else:
                html.append("<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            html.append(escape(line))
            continue
        if not line:
            flush_paragraph(); close_list()
            if table:
                html.append("</table>"); table = False; table_header_written = False
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph(); close_list()
            level = len(heading.group(1))
            title = inline(heading.group(2), source, target)
            if level >= 2:
                anchor = f"section-{len(toc) + 1}"
                toc.append((level, title, anchor))
                html.append(f'<h{level} id="{anchor}">{title}</h{level}>')
            else:
                html.append(f'<h{level}>{title}</h{level}>')
            continue
        if line.startswith("|"):
            flush_paragraph(); close_list()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
                continue
            if not table: html.append("<table>"); table = True; table_header_written = False
            tag = "th" if not table_header_written else "td"
            table_header_written = True
            html.append("<tr>" + "".join(f"<{tag}>{inline(cell, source, target)}</{tag}>" for cell in cells) + "</tr>")
            continue
        item = re.match(r"^\s*[-*+]\s+(.+)$", line)
        ordered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if item or ordered:
            wanted = "ol" if ordered else "ul"
            if list_tag != wanted:
                close_list()
                html.append(f"<{wanted}>"); list_tag = wanted
            html.append(f"<li>{inline((ordered or item).group(1), source, target)}</li>")
            continue
        if list_tag:
            close_list()
        if line.startswith("> "):
            html.append(f"<blockquote>{inline(line[2:], source, target)}</blockquote>")
        else:
            paragraph.append(line)
    flush_paragraph(); close_list()
    if table: html.append("</table>")
    if in_code: html.append("</code></pre>")
    return "\n".join(html), toc


def page(title, content, language, index_href, home_href, toc=(), switch_href=None):
    lang_name = "zh-Hans" if language == "zh" else "en"
    home = home_href
    if language == "zh":
        home_label, docs_label, docs_title, github = "首页", "文档", "文档", "GitHub"
        switch = f'<a href="{switch_href or "../../docs/"}">English</a>'
    else:
        home_label, docs_label, docs_title, github = "Home", "Docs", "Documentation", "GitHub"
        switch = f'<a href="{switch_href or "../zh-Hans/docs/"}">中文</a>'
    toc_html = "".join(f'<li class="toc-level-{level}"><a href="#{anchor}">{label}</a></li>' for level, label, anchor in toc)
    sidebar = f'<aside><p class="current-doc">{escape(title)}</p><ul class="toc">{toc_html}</ul></aside>' if toc else f'<aside><a href="{index_href}">{docs_title}</a></aside>'
    return f'''<!doctype html><html lang="{lang_name}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="{escape(title)} — IronMLX documentation"><link rel="icon" href="{home}assets/favicon.png"><link rel="stylesheet" href="{home}styles.css"><title>{escape(title)} — IronMLX</title></head><body><header class="nav"><a class="brand" href="{home}"><span class="mark">Fe</span><span>IronMLX</span></a><nav><a href="{home}">{home_label}</a><a href="{index_href}">{docs_label}</a>{switch}</nav></header><main class="docs-layout">{sidebar}<article class="doc-content">{content}</article></main><footer><span>© IronMLX</span><span><a href="https://github.com/apepkuss/ironmlx">{github}</a></span></footer></body></html>'''


def build(language, output, docs_root):
    output.mkdir(parents=True, exist_ok=True)
    index_href = "./" if language == "en" else "./"
    links = []
    for filename, en_title, zh_title in PAGES:
        source = docs_root / ("zh-CN" if language == "zh" else "") / filename
        if not source.is_file():
            raise SystemExit(f"missing documentation source: {source}")
        title = zh_title if language == "zh" else en_title
        target = output / Path(filename).with_suffix(".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        content, toc = render(source.read_text(encoding="utf-8"), source, target)
        website_root = ROOT / "website"
        home_href = Path(__import__("os").path.relpath(website_root, target.parent)).as_posix() + "/"
        counterpart_root = website_root / ("docs" if language == "zh" else "zh-Hans/docs")
        counterpart = counterpart_root / Path(filename).with_suffix(".html")
        switch_href = Path(__import__("os").path.relpath(counterpart, target.parent)).as_posix()
        target.write_text(page(title, content, language, index_href, home_href, toc, switch_href), encoding="utf-8")
        links.append(f'<li><a href="{target.relative_to(output).as_posix()}">{escape(title)}</a></li>')
    heading = "<h1>文档</h1>" if language == "zh" else "<h1>Documentation</h1>"
    home_href = "../../" if language == "zh" else "../"
    (output / "index.html").write_text(page("文档" if language == "zh" else "Documentation", heading + f'<ul>{"".join(links)}</ul>', language, index_href, home_href), encoding="utf-8")


assets = ROOT / "website" / "assets"
assets.mkdir(parents=True, exist_ok=True)
for image in (ROOT / "docs" / "images").glob("*"):
    if image.is_file():
        shutil.copy2(image, assets / image.name)
for directory in (OUT, ZH_OUT):
    if directory.exists(): shutil.rmtree(directory)
build("en", OUT, ROOT / "docs")
build("zh", ZH_OUT, ROOT / "docs")
