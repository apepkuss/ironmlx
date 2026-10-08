#!/usr/bin/env python3
"""Render maintained Markdown into the bilingual, dependency-free docs website."""
from html import escape, unescape
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
WEBSITE = ROOT / "website"
PAGES = (
    ("user-guide.md", "User guide", "用户指南", "Install the app, import a model, and make your first request.", "安装应用、导入模型，完成第一次请求。"),
    ("developer-guide.md", "Developer guide", "开发者指南", "Find API examples, CLI configuration, and source development resources.", "查找 API 示例、CLI 配置与源码开发入口。"),
    ("supported-models.md", "Supported models", "支持的模型", "Compare model names, weight formats, and capabilities.", "对照具体模型、权重格式与支持能力。"),
    ("api-reference.md", "API reference", "API 参考", "Find every documented endpoint and choose an API topic.", "查找全部已公开文档的端点，按能力进入 API 专题。"),
    ("service-api.md", "Service and management API", "服务与管理 API", "Look up shared access conventions, health, model discovery, and management endpoints.", "查阅公共访问约定、健康检查、模型发现与管理接口。"),
    ("text-vision-api.md", "Text and vision API", "文本与视觉 API", "Use Responses, Chat Completions, and Messages for text and image understanding.", "查阅文本与图片理解、思考、工具、结构化输出和流式协议。"),
    ("text-embeddings.md", "Embedding API", "Embedding API", "Create text, image, audio, and combined vectors with EmbeddingGemma 2.", "通过 EmbeddingGemma 2 生成文本、图片、音频与组合向量。"),
    ("audio-speech-api.md", "Speech synthesis API", "语音合成 API", "Generate speech and manage voices through the local API.", "通过本地 API 合成语音与管理声音配置。"),
    ("image-generation-api.md", "Image generation API", "图片生成 API", "Generate images and edit a reference image through the local API.", "通过本地 API 生成图片与编辑参考图片。"),
    ("laya-systemone-api.md", "System One API", "System One API", "Send choice, score, and true/false probability requests to Laya.", "通过 Laya 处理选择、评分与真假概率请求。"),
    ("building-from-source.md", "Build from source", "从源码构建", "Set up your environment and build IronMLX.", "准备开发环境，从源码构建 IronMLX。"),
    ("audio-library.md", "ironmlx-audio developer reference", "ironmlx-audio 开发参考", "Review the audio crate interfaces, resources, and integration contracts.", "查阅音频 crate 的接口、资源及集成契约。"),
    ("contributing.md", "Contributing", "参与开发", "Review contribution terms and required verification.", "了解贡献条款与必要验证。"),
    ("hermes-agent.md", "Hermes Agent", "Hermes Agent", "Connect Hermes Agent to your local inference service.", "将 Hermes Agent 接入本地推理服务。"),
    ("oh-my-pi.md", "oh-my-pi", "oh-my-pi", "Configure oh-my-pi to use IronMLX.", "配置 oh-my-pi 使用 IronMLX。"),
    ("dsh.md", "DeepSeek Harness (DSH)", "DeepSeek Harness（DSH）", "Connect DSH CLI and DSH Desktop to IronMLX.", "将 DSH CLI 与 DSH Desktop 接入 IronMLX。"),
    ("dflash2-server-api.md", "DFlash2 configuration and usage", "DFlash2 配置与使用", "Configure DFlash2 for CLI generation, HTTP serving, or the App.", "通过 CLI 或 App 配置 DFlash2 加速与推理服务。"),
    ("mtp-server-api.md", "Qwen MTP configuration and usage", "Qwen MTP 配置与使用", "Enable Qwen MTP in CLI generation and HTTP serving.", "配置 Qwen MTP 的 CLI 生成与 HTTP 服务，了解请求限制。"),
    ("engine-pool.md", "CLI multi-model serving (EnginePool)", "CLI 多模型服务（EnginePool）", "Configure model routing and loading with a CLI manifest.", "通过 CLI 配置多模型路由、加载与卸载。"),
    ("scheduler-profile-v5.md", "Scheduler performance calibration", "调度性能校准", "Run optional offline calibration for model-specific scheduling.", "为模型执行可选的离线调度校准，日常使用无需配置。"),
    ("automatic-updates.md", "Automatic updates", "自动更新", "Understand update checks and installation behavior.", "了解更新检查与安装行为。"),
    ("diagnostic-bundle.md", "Diagnostic export", "诊断信息导出", "Export diagnostic information to investigate problems.", "导出诊断信息，辅助定位问题。"),
    ("known-issues.md", "Known issues", "已知问题", "Review current limitations and workarounds.", "查看当前限制与解决办法。"),
    ("troubleshooting.md", "Troubleshooting", "故障排查", "Resolve common installation and runtime problems.", "定位常见的安装与运行问题。"),
    ("storage-and-uninstall.md", "Storage and uninstall", "数据位置与卸载", "Find local data and remove the app when needed.", "查找本地数据目录与卸载方法。"),
    ("support.md", "Support", "获取支持", "Find reporting guidance and the supported platform.", "了解支持平台与问题反馈方式。"),
    ("versioning-and-releases.md", "Versioning and releases", "版本与发布", "Understand version numbers and release channels.", "了解版本编号与发布渠道。"),
    ("release-notes/0.2.0.md", "0.2.0 release notes", "0.2.0 发布说明", "Review the changes in version 0.2.0.", "查看 0.2.0 版本的变更。"),
    ("release-notes/0.1.0.md", "0.1.0 release notes", "0.1.0 发布说明", "Review the changes in version 0.1.0.", "查看 0.1.0 版本的变更。"),
    ("security-boundary.md", "Security boundary", "安全边界", "Understand local access and network exposure.", "了解本地访问与网络暴露边界。"),
    ("security.md", "Security policy", "安全漏洞报告", "Report vulnerabilities privately and review support policies.", "私密报告漏洞，了解安全支持政策。"),
    ("privacy.md", "Privacy", "隐私说明", "Understand how the app handles your data.", "了解应用如何处理你的数据。"),
    ("model-license-boundary.md", "Model rights boundary", "模型权利边界", "Review model licensing and usage responsibilities.", "了解模型许可与使用责任。"),
)
GROUPS = (
    ("start", "Getting started", "开始使用", "Everything you need for your first local inference.", "从安装到首次请求，开始本地推理。", PAGES[:3]),
    ("api", "Development & API", "开发与 API", "Integrate APIs, build from source, and contribute to IronMLX.", "接入 API、从源码构建与参与 IronMLX 开发。", PAGES[3:13]),
    ("agents", "Supported agents", "支持的 Agents", "Configure supported agent applications to use local IronMLX inference.", "配置受支持的 Agent 应用，连接 IronMLX 本地推理服务。", PAGES[13:16]),
    ("inference", "CLI & advanced configuration", "CLI 与高级配置", "Configure CLI serving, acceleration, and optional offline calibration.", "配置 CLI 服务、生成加速与可选的离线性能校准。", PAGES[16:20]),
    ("maintenance", "Maintenance & troubleshooting", "维护与排障", "Keep the app running and investigate problems.", "维护应用运行，诊断与处理问题。", PAGES[20:26]),
    ("releases", "Releases", "发布", "Review versions, release channels, and release notes.", "查看版本信息、发布渠道与发布说明。", PAGES[26:29]),
    ("security", "Security & licensing", "安全与许可", "Understand privacy, access, and model usage boundaries.", "了解隐私、访问权限与模型使用边界。", PAGES[29:]),
)
PAGE_MAP = {entry[0]: entry for entry in PAGES}
GROUP_MAP = {entry[0]: group for group in GROUPS for entry in group[5]}
PAGE_PARENT = {filename: "api-reference.md" for filename in (
    "service-api.md", "text-vision-api.md", "text-embeddings.md", "audio-speech-api.md",
    "image-generation-api.md", "laya-systemone-api.md",
)}
CHILD_PAGES = {"api-reference.md": tuple(PAGE_MAP[filename] for filename in PAGE_PARENT)}

# Publish repository-level sources under stable bilingual documentation URLs.
SOURCE_OVERRIDES = {
    "audio-library.md": {"en": "ironmlx-audio/README.md", "zh": "ironmlx-audio/README.zh-CN.md"},
    "support.md": {"en": "SUPPORT.md", "zh": "docs/zh-CN/support.md"},
    "contributing.md": {"en": "CONTRIBUTING.md", "zh": "docs/zh-CN/contributing.md"},
    "security.md": {"en": "SECURITY.md", "zh": "docs/zh-CN/security.md"},
}


def source_path(filename, language):
    if filename in SOURCE_OVERRIDES:
        return ROOT / SOURCE_OVERRIDES[filename][language]
    return ROOT / "docs" / ("zh-CN" if language == "zh" else "") / filename


SOURCE_MAP = {
    source_path(entry[0], language).resolve(): (entry[0], language)
    for entry in PAGES for language in ("en", "zh")
}


def localized(en, zh, language):
    return zh if language == "zh" else en


def relative(path, parent):
    return Path(os.path.relpath(path, parent)).as_posix()


def slug(text):
    text = re.sub(r"!?\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_]", "", text).lower()
    return re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", text)).strip("-") or "section"


def inline(text, source, target):
    protected = []

    def protect(match):
        protected.append(f"<code>{escape(match.group(1))}</code>")
        return f"\x00{len(protected) - 1}\x00"

    text = escape(re.sub(r"`([^`]+)`", protect, text), quote=False)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)

    def image(match):
        alt, href = unescape(match.group(1)), unescape(match.group(2))
        if href.startswith(("images/", "../images/")):
            href = relative(WEBSITE / "assets" / Path(href).name, target.parent)
        return f'<img src="{escape(href, quote=True)}" alt="{escape(alt, quote=True)}" loading="lazy">'

    text = re.sub(r"!\[([^]]*)\]\(([^)]+)\)", image, text)

    def link(match):
        label, href = match.group(1), unescape(match.group(2))
        path, separator, fragment = href.partition("#")
        if path.endswith(".md") and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", path):
            source_target = (source.parent / path).resolve()
            published = SOURCE_MAP.get(source_target)
            if published:
                filename, source_language = published
                # Root documents are shared canonical references; use the reader's language.
                if filename in SOURCE_OVERRIDES and not source_target.is_relative_to(ROOT / "docs"):
                    source_language = "zh" if target.is_relative_to(WEBSITE / "zh-Hans") else "en"
                generated_root = WEBSITE / "zh-Hans" / "docs" if source_language == "zh" else WEBSITE / "docs"
                href = relative((generated_root / filename).with_suffix(".html"), target.parent)
            elif source_target.is_relative_to(ROOT):
                href = f"https://github.com/apepkuss/ironmlx/blob/dev/{source_target.relative_to(ROOT).as_posix()}"
            if separator:
                href += "#" + fragment
        elif path and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", path):
            resource = (source.parent / path).resolve()
            if resource.is_file() and resource.is_relative_to(ROOT) and not resource.is_relative_to(WEBSITE):
                href = f"https://github.com/apepkuss/ironmlx/blob/dev/{resource.relative_to(ROOT).as_posix()}"
                if separator:
                    href += "#" + fragment
        return f'<a href="{escape(href, quote=True)}">{label}</a>'

    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", link, text)
    return re.sub(r"\x00(\d+)\x00", lambda match: protected[int(match.group(1))], text)


def render(markdown, source, target):
    html, toc, sections = [], [], []
    paragraph, quote, code = [], [], []
    in_code, list_tag, table = False, None, False
    language = "text"
    anchors = {}
    section_text = []

    def flush_paragraph():
        if paragraph:
            html.append(f"<p>{inline(' '.join(paragraph), source, target)}</p>")
            paragraph.clear()

    def close_list():
        nonlocal list_tag
        if list_tag:
            html.append(f"</{list_tag}>")
            list_tag = None

    def close_table():
        nonlocal table
        if table:
            html.append("</tbody></table></div>")
            table = False

    def flush_quote():
        if quote:
            content = " ".join(quote)
            kind = "note"
            marker = re.match(r"\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*", content, re.I)
            if marker:
                kind = "warning" if marker.group(1).upper() in {"WARNING", "CAUTION"} else "note"
                content = content[marker.end():]
            html.append(f'<blockquote class="callout callout-{kind}">{inline(content, source, target)}</blockquote>')
            quote.clear()

    def flush_code():
        if language == "mermaid":
            diagram_source = "\n".join(code).strip()
            digest = sha256(diagram_source.encode("utf-8")).hexdigest()
            asset = WEBSITE / "assets" / f"mermaid-{digest}.svg"
            if not asset.is_file():
                raise ValueError(f"Render the Mermaid source to {asset.relative_to(ROOT)} before building")
            description = "音频库依赖关系" if target.is_relative_to(WEBSITE / "zh-Hans") else "Audio library dependencies"
            html.append(f'<figure class="doc-diagram"><img src="{relative(asset, target.parent)}" alt="{description}" loading="lazy"></figure>')
            code.clear()
            return
        copy = "复制" if target.is_relative_to(WEBSITE / "zh-Hans") else "Copy"
        safe_language = escape(language, quote=True)
        html.append(f'<div class="code-block"><div class="code-toolbar"><span>{safe_language}</span><button type="button" data-copy-code>{copy}</button></div><pre><code class="language-{safe_language}">{escape(chr(10).join(code))}</code></pre></div>')
        code.clear()

    def flush_section():
        if sections:
            sections[-1]["text"] = " ".join(section_text)
        section_text.clear()

    lines = []
    fenced = False
    for raw in markdown.splitlines():
        if raw.startswith("```"):
            fenced = not fenced
        is_item = re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", raw)
        if not fenced and raw[:1].isspace() and not is_item and not raw.lstrip().startswith("|") and lines and re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", lines[-1]):
            lines[-1] += " " + raw.strip()
        else:
            lines.append(raw)

    for raw in lines:
        line = raw.rstrip()
        # The header already switches language while preserving the current page.
        if not in_code:
            if re.fullmatch(r"<!--.*-->", line.strip()):
                continue
            line = re.sub(r"^\[(?:English|简体中文|中文)\]\([^)]+\.md\)(?:\s*·\s*)?", "", line)
        if line.startswith("```"):
            flush_paragraph()
            flush_quote()
            close_list()
            close_table()
            if in_code:
                flush_code()
            else:
                language = re.sub(r"[^a-zA-Z0-9_+-]", "", line[3:].strip()) or "text"
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            section_text.append(line)
            continue
        if line.startswith(">"):
            flush_paragraph()
            close_list()
            close_table()
            quote.append(line.lstrip(">").strip())
            section_text.append(line)
            continue
        flush_quote()
        if not line.strip():
            flush_paragraph()
            close_list()
            close_table()
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_list()
            close_table()
            level, raw_title = len(heading.group(1)), heading.group(2)
            title = inline(raw_title, source, target)
            if level >= 2:
                flush_section()
                base = slug(raw_title)
                count = anchors.get(base, 0)
                anchors[base] = count + 1
                anchor = f"{base}-{count}" if count else base
                toc.append((level, title, anchor))
                sections.append({"title": re.sub(r"[`*_]", "", raw_title), "anchor": anchor, "text": ""})
                # Preserve links to the anchors emitted by the previous renderer.
                html.append(f'<span id="section-{len(toc)}" class="legacy-anchor"></span><h{level} id="{anchor}">{title}<a class="heading-anchor" href="#{anchor}" aria-label="{escape(raw_title, quote=True)}">#</a></h{level}>')
            else:
                html.append(f"<h1>{title}</h1>")
            continue
        section_text.append(line)
        if line.lstrip().startswith("|"):
            flush_paragraph()
            close_list()
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
                continue
            if not table:
                html.append('<div class="doc-table" role="region" aria-label="' + ("表格" if target.is_relative_to(WEBSITE / "zh-Hans") else "Table") + '" tabindex="0"><table><thead><tr>' + "".join(f"<th>{inline(cell, source, target)}</th>" for cell in cells) + "</tr></thead><tbody>")
                table = True
            else:
                html.append("<tr>" + "".join(f"<td>{inline(cell, source, target)}</td>" for cell in cells) + "</tr>")
            continue
        close_table()
        item = re.match(r"^\s*[-*+]\s+(.+)$", line)
        ordered = re.match(r"^\s*(\d+)[.)]\s+(.+)$", line)
        if item or ordered:
            flush_paragraph()
            wanted = "ol" if ordered else "ul"
            if list_tag != wanted:
                close_list()
                start = f' start="{int(ordered.group(1))}"' if ordered and int(ordered.group(1)) != 1 else ""
                html.append(f"<{wanted}{start}>")
                list_tag = wanted
            body = ordered.group(2) if ordered else item.group(1)
            html.append(f"<li>{inline(body, source, target)}</li>")
            continue
        close_list()
        if re.fullmatch(r"(?:-{3,}|\*{3,})", line):
            flush_paragraph()
            html.append("<hr>")
        else:
            paragraph.append(line)
    flush_paragraph()
    flush_quote()
    close_list()
    close_table()
    if in_code:
        flush_code()
    flush_section()
    return "\n".join(html), toc, sections


def icon(name):
    paths = {
        "start": '<path d="M12 3 3 8l9 5 9-5-9-5ZM3 12l9 5 9-5M3 16l9 5 9-5"/>',
        "api": '<path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-13-2 20"/>',
        "agents": '<rect x="4" y="7" width="16" height="13" rx="3"/><path d="M12 3v4M8 12h.01M16 12h.01M8 16h8M1 11v5m22-5v5"/>',
        "inference": '<rect x="5" y="5" width="14" height="14" rx="3"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/>',
        "maintenance": '<path d="M14 6a5 5 0 0 0-6 6L3 17a3 3 0 0 0 4 4l5-5a5 5 0 0 0 6-6l-4 2-2-2 2-4Z"/>',
        "releases": '<path d="M5 4h10l4 4v12H5V4Zm10 0v5h4M8 13h8m-8 4h5"/>',
        "security": '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm-4 9 3 3 5-6"/>',
        "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
        "arrow": '<path d="M4 12h16m-6-6 6 6-6 6"/>',
        "menu": '<path d="M4 6h16M4 12h16M4 18h16"/>',
        "close": '<path d="m6 6 12 12M6 18 18 6"/>',
    }
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'


def sidebar(language, output, target, current):
    label = localized("Documentation", "文档", language)
    overview = localized("Overview", "文档概览", language)
    active = ' aria-current="page"' if current is None else ""
    result = [f'<nav aria-label="{label}"><a class="sidebar-overview" href="{relative(output / "index.html", target.parent)}"{active}>{overview}</a>']

    def entry_link(entry):
        active = ' aria-current="page"' if current == entry[0] else ""
        href = relative((output / entry[0]).with_suffix(".html"), target.parent)
        return f'<a href="{href}"{active}>{localized(entry[1], entry[2], language)}</a>'

    for group in GROUPS:
        result.append(f'<section class="sidebar-group"><h2>{localized(group[1], group[2], language)}</h2><ul>')
        for entry in group[5]:
            if entry[0] in PAGE_PARENT:
                continue
            result.append(f'<li>{entry_link(entry)}')
            if entry[0] in CHILD_PAGES:
                result.append('<ul class="sidebar-children">')
                result.extend(f'<li>{entry_link(child)}</li>' for child in CHILD_PAGES[entry[0]])
                result.append('</ul>')
            result.append('</li>')
        result.append("</ul></section>")
    result.append("</nav>")
    return "".join(result)


def nav_icon(name, home):
    return f'<svg class="nav-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><use href="{home}assets/navigation-icons.svg#{name}"></use></svg>'


def theme_picker(language):
    system = '<svg class="theme-option-icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="14" rx="2"></rect><path d="M8 22h8M12 18v4"></path></svg>'
    sun = '<svg class="theme-option-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42"></path></svg>'
    moon = '<svg class="theme-option-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.4 14.6A8.5 8.5 0 0 1 9.4 3.6a8.5 8.5 0 1 0 11 11Z"></path></svg>'
    options = (("system", system, "System", "跟随系统"), ("light", sun, "Light", "浅色"), ("dark", moon, "Dark", "深色"))
    items = "".join(f'<button type="button" role="menuitemradio" data-theme-option="{value}">{symbol}<span>{localized(en, zh, language)}</span><span class="theme-check" aria-hidden="true">✓</span></button>' for value, symbol, en, zh in options)
    trigger = sun.replace('class="theme-option-icon"', 'class="theme-icon"')
    return f'<div class="theme-picker" data-theme-picker data-theme-label="{localized("Theme", "主题", language)}"><button class="theme-trigger" type="button" aria-label="{localized("Theme", "主题", language)}" aria-haspopup="menu" aria-expanded="false" data-theme-trigger>{trigger}</button><div class="theme-menu" role="menu" data-theme-menu hidden>{items}</div></div>'


def page(title, content, language, output, target, toc=(), current=None):
    lang_name = "zh-Hans" if language == "zh" else "en"
    home = relative(WEBSITE, target.parent) + "/"
    local_home = relative(WEBSITE / "zh-Hans" if language == "zh" else WEBSITE, target.parent) + "/"
    index = relative(output / "index.html", target.parent)
    counterpart_root = WEBSITE / "docs" if language == "zh" else WEBSITE / "zh-Hans" / "docs"
    switch = relative(counterpart_root / target.relative_to(output), target.parent)
    search = localized("Search documentation", "搜索文档", language)
    docs_title = localized("Documentation", "文档", language)
    skip = localized("Skip to content", "跳转到正文", language)
    on_page = localized("On this page", "本页目录", language)
    nav_label = localized("Browse documentation", "浏览文档", language)
    close = localized("Close", "关闭", language)
    home_label = localized("Home", "首页", language)
    toc_html = "".join(f'<li class="toc-level-{level}"><a href="#{anchor}">{label}</a></li>' for level, label, anchor in toc if level <= 3)
    outline = f'<aside class="docs-outline" aria-label="{on_page}"><p>{on_page}</p><ul>{toc_html}</ul></aside>' if toc_html else ""
    mobile_outline = f'<details class="mobile-outline"><summary>{on_page}</summary><ul>{toc_html}</ul></details>' if toc_html else ""
    nav_html = sidebar(language, output, target, current)
    breadcrumbs = ""
    if current:
        group = GROUP_MAP[current]
        breadcrumbs = f'<div class="doc-breadcrumbs"><a href="{index}">{docs_title}</a><span aria-hidden="true">/</span><a href="{index}#{group[0]}">{localized(group[1], group[2], language)}</a></div>'
        if current in PAGE_PARENT:
            parent = PAGE_MAP[PAGE_PARENT[current]]
            href = relative((output / parent[0]).with_suffix(".html"), target.parent)
            breadcrumbs = f'<div class="doc-breadcrumbs"><a href="{index}">{docs_title}</a><span aria-hidden="true">/</span><a href="{href}">{localized(parent[1], parent[2], language)}</a><span aria-hidden="true">/</span><span aria-current="page">{escape(title)}</span></div>'
    search_index = relative(output / "search-index.json", target.parent)
    homepage_class = " docs-overview" if not current else ""
    body_class = "docs-page docs-models-page" if current == "supported-models.md" else "docs-page"
    if current == "api-reference.md":
        body_class += " docs-api-reference-page"
    return f'''<!doctype html>
<html lang="{lang_name}" data-theme="system">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark"><meta name="description" content="{escape(title, quote=True)} — IronMLX documentation">
<link rel="icon" href="{home}assets/favicon.png"><title>{escape(title)} — IronMLX</title>
<script src="{home}theme.js"></script><link rel="stylesheet" href="{home}styles.css"><link rel="stylesheet" href="{home}docs.css">
<script src="{home}docs.js" defer></script>
</head>
<body class="{body_class}" data-doc-language="{language}" data-search-index="{search_index}">
<a class="skip-link" href="#doc-main">{skip}</a>
<header class="docs-header"><div class="docs-header-inner">
<a class="brand" href="{local_home}"><span class="mark">Fe</span><span>IronMLX</span></a><a class="docs-header-label" href="{index}">{docs_title}</a>
<button class="docs-search-trigger" type="button" data-open-search>{icon("search")}<span>{search}</span><kbd data-search-shortcut>⌘ K</kbd></button>
<nav class="docs-header-links" aria-label="{localized("Site navigation", "网站导航", language)}"><a class="nav-link" href="{local_home}">{nav_icon("home", home)}<span>{home_label}</span></a><a class="nav-link" href="https://github.com/apepkuss/ironmlx">{nav_icon("github", home)}<span>GitHub</span></a><a class="nav-link" href="{switch}" lang="{ 'en' if language == 'zh' else 'zh-Hans'}">{nav_icon("language", home)}<span>{localized("中文", "English", language)}</span></a>{theme_picker(language)}</nav>
<button type="button" class="docs-menu-trigger" aria-label="{nav_label}" aria-haspopup="dialog" data-open-navigation>{icon("menu")}</button>
</div></header>
<div class="docs-layout{homepage_class}"><aside class="docs-sidebar">{nav_html}</aside><main id="doc-main" class="doc-main">{breadcrumbs}{mobile_outline}<article class="doc-content">{content}</article></main>{outline}</div>
<footer class="docs-footer"><span>IronMLX · {localized("Local AI on Apple Silicon", "Apple Silicon 本地 AI", language)}</span><a href="https://github.com/apepkuss/ironmlx">GitHub ↗</a></footer>
<dialog class="docs-search-dialog" aria-labelledby="search-label"><div class="search-input-row">{icon("search")}<label class="sr-only" id="search-label" for="docs-search-input">{search}</label><input id="docs-search-input" type="search" placeholder="{search}…" autocomplete="off" spellcheck="false"><button type="button" class="dialog-close" aria-label="{close}" data-close-search>{icon("close")}</button></div><p class="search-status" aria-live="polite"></p><div class="search-results"></div><div class="search-footer"><span>{localized("↑ ↓ to navigate · Enter to open", "↑ ↓ 选择 · Enter 打开", language)}</span><span>Esc {close}</span></div></dialog>
<dialog class="docs-navigation-dialog" aria-labelledby="navigation-label"><div class="navigation-dialog-heading"><strong id="navigation-label">{nav_label}</strong><button type="button" class="dialog-close" aria-label="{close}" data-close-navigation>{icon("close")}</button></div>{nav_html}</dialog>
<div class="sr-only" aria-live="polite" data-copy-status></div>
</body></html>'''


def overview(language):
    title = localized("IronMLX documentation", "IronMLX 文档", language)
    lead = localized("Run models locally. Connect your tools. Make the most of your Mac.", "在 Mac 上运行本地模型，连接你的工具，掌握推理服务。", language)
    cards = (
        ("start", "user-guide.md", "Start using IronMLX", "开始使用", "Install the app, download models, and start local inference.", "安装应用、下载模型，开始本地推理。"),
        ("api", "developer-guide.md", "Developer guide", "开发者指南", "Connect to APIs, use the CLI, or develop from source.", "接入 API、使用 CLI，或从源码开发。"),
        ("maintenance", "troubleshooting.md", "Troubleshooting", "故障排查", "Resolve installation, model loading, and runtime problems.", "定位安装、模型加载与运行问题。"),
    )
    html = [f'<div class="docs-intro"><p class="docs-eyebrow">{localized("GUIDES & REFERENCE", "指南与参考", language)}</p><h1>{title}</h1><p class="docs-lead">{lead}</p></div><div class="docs-entry-grid">']
    for symbol, filename, en_title, zh_title, en_desc, zh_desc in cards:
        href = Path(filename).with_suffix(".html").as_posix()
        html.append(f'<a class="docs-entry" href="{href}"><span class="docs-entry-icon">{icon(symbol)}</span><h2>{localized(en_title, zh_title, language)}{icon("arrow")}</h2><p>{localized(en_desc, zh_desc, language)}</p></a>')
    html.append('</div><div class="docs-categories">')
    for group in GROUPS:
        html.append(f'<section class="docs-category" id="{group[0]}"><h2>{icon(group[0])}{localized(group[1], group[2], language)}</h2><p class="category-description">{localized(group[3], group[4], language)}</p><ul>')
        for entry in group[5]:
            if entry[0] in PAGE_PARENT:
                continue
            href = Path(entry[0]).with_suffix(".html").as_posix()
            html.append(f'<li><a href="{href}"><span class="category-link-title">{localized(entry[1], entry[2], language)}<span aria-hidden="true">↗</span></span><span class="category-link-description">{localized(entry[3], entry[4], language)}</span></a></li>')
        html.append("</ul></section>")
    html.append("</div>")
    return "\n".join(html)


def article_footer(language, entry, source, output, target):
    result = subprocess.run(["git", "log", "-1", "--format=%cs", "--", source.relative_to(ROOT).as_posix()], cwd=ROOT, capture_output=True, text=True, check=True)
    updated = result.stdout.strip()
    edit_url = f"https://github.com/apepkuss/ironmlx/edit/dev/{source.relative_to(ROOT).as_posix()}"
    date = f'<span>{localized("Last updated", "最后更新", language)} <time datetime="{updated}">{updated}</time></span>' if updated else ""
    html = [f'<div class="doc-edit-row"><a href="{edit_url}">{localized("Edit this page on GitHub", "在 GitHub 上编辑此页", language)} ↗</a>{date}</div><nav class="doc-pagination" aria-label="{localized("Adjacent pages", "相邻文档", language)}">']
    index = PAGES.index(entry)
    for position, label, arrow in ((index - 1, localized("Previous", "上一篇", language), "←"), (index + 1, localized("Next", "下一篇", language), "→")):
        if 0 <= position < len(PAGES):
            neighbor = PAGES[position]
            href = relative((output / neighbor[0]).with_suffix(".html"), target.parent)
            html.append(f'<a href="{href}"><span>{label} {arrow}</span><strong>{localized(neighbor[1], neighbor[2], language)}</strong></a>')
        else:
            html.append("<span></span>")
    html.append("</nav>")
    return "".join(html)


def build(language):
    output = WEBSITE / "zh-Hans" / "docs" if language == "zh" else WEBSITE / "docs"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    search_index = []
    for entry in PAGES:
        source = source_path(entry[0], language)
        if not source.is_file():
            raise SystemExit(f"missing documentation source: {source}")
        title = localized(entry[1], entry[2], language)
        target = (output / entry[0]).with_suffix(".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        markdown = source.read_text(encoding="utf-8")
        content, toc, sections = render(markdown, source, target)
        content += article_footer(language, entry, source, output, target)
        target.write_text(page(title, content, language, output, target, toc, entry[0]), encoding="utf-8")
        group = GROUP_MAP[entry[0]]
        search_group = localized(group[1], group[2], language)
        if entry[0] in PAGE_PARENT:
            parent = PAGE_MAP[PAGE_PARENT[entry[0]]]
            search_group = localized(parent[1], parent[2], language)
        search_index.append({"title": title, "group": search_group, "description": localized(entry[3], entry[4], language), "href": target.relative_to(output).as_posix(), "text": markdown, "sections": sections})
    (output / "search-index.json").write_text(json.dumps(search_index, ensure_ascii=False), encoding="utf-8")
    target = output / "index.html"
    target.write_text(page(localized("Documentation", "文档", language), overview(language), language, output, target), encoding="utf-8")


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/build-supported-models-docs.py")], check=True)
    assets = WEBSITE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for image in (ROOT / "docs" / "images").glob("*"):
        if image.is_file():
            shutil.copy2(image, assets / image.name)
    for language in ("en", "zh"):
        build(language)
    print(f"Built {len(PAGES) * 2} articles, two overviews, and bilingual search indexes.")


if __name__ == "__main__":
    main()
