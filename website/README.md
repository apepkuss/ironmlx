# IronMLX website

Static landing page and generated user documentation for IronMLX, designed for GitHub Pages.
Build the documentation pages from the maintained Markdown sources, then preview locally:

```bash
python3 scripts/build-website-docs.py
python3 -m http.server 8000
```

Then open <http://localhost:8000/website/> from the repository root.

The theme control defaults to the system appearance. Explicit light or dark
choices are stored locally in the browser and shared by the landing page and
generated documentation pages.

The documentation shell is built by `scripts/build-website-docs.py`. Its `PAGES`
and `GROUPS` define the bilingual directory, descriptions, and reading order.
Update those entries when adding a published page. Markdown remains the source
of truth; generated HTML and search indexes are ignored by Git.

Getting started links the user guide, developer guide, and supported models.
The developer guide includes the API quick start and directs readers to API,
CLI, source development, and maintenance references. The overview cards link
to the user guide, developer guide, and troubleshooting page. The developer
guide card opens the page from the beginning and links to the API reference hub.
The hub provides six topic entry points and an index of all documented endpoints.
Service and management, text and vision, text/image/audio embeddings, speech
synthesis, image generation, and System One are child pages in both desktop
and mobile navigation.

Development & API groups the API hub with source builds, library references,
and contribution guidance. Releases lists version information, release notes
from newest to oldest. The bilingual release pipeline documents remain in the
repository for release maintainers and are excluded from the public website.
Overview, sidebar, search,
and adjacent-page navigation use the same grouping and reading order.
`PAGE_PARENT` and `CHILD_PAGES` define API topic nesting; child pages have an
API reference breadcrumb and search group, while overview categories show the
hub. `api-reference.html` serves the hub; text and vision contracts live at
`text-vision-api.html`. Update Markdown links when moving contract sections.
Shared addresses, authentication, health checks, model discovery and management
contracts live in the service and management reference. Each inference topic
maintains its own protocol fields, examples, errors and limits.
SDK checks live in the source build guide and API maintenance requirements
live in the contribution guide.

`SOURCE_OVERRIDES` maps stable page names to canonical sources outside `docs/`,
including the audio library README and the root support, contribution, and
security documents. Each page has a corresponding Chinese source. Source links
resolve to these published pages regardless of the original file location;
GitHub edit links continue to use the canonical source path.

Mermaid diagrams use checked-in SVGs under `assets/mermaid-<sha256>.svg` so the
build and reading experience need no external renderer. The hash is computed
from the UTF-8 Mermaid block with leading/trailing whitespace stripped. When
changing a diagram, render and verify the updated Mermaid source, then save the
SVG under its new hash; the build rejects diagrams without a matching asset.

`docs.css` contains the responsive documentation layout. `docs.js` provides local
full-text search (loaded on demand), keyboard navigation, code highlighting and
copying, and the current-section indicator. Search needs no external service or
build dependency. The shared `theme.js` controls appearance across the website.

Preview the documentation directly at:

- English: <http://localhost:8000/website/docs/>
- 简体中文: <http://localhost:8000/website/zh-Hans/docs/>

Article links use stable heading anchors and preserve the previous `section-N`
anchors. Links to Markdown pages outside the published subset open their source
on GitHub. The header language control opens the corresponding translated page,
including nested release notes.

The concise `docs/supported-models.md` lists are generated in both languages from
the App's `supported-models.json` by `scripts/build-supported-models-docs.py`.
The website build refreshes them automatically. After editing the App catalogue,
run that generator and commit its Markdown output; `--check` detects stale lists.
Model groups and variants retain the App's names, classification, capabilities,
and weight formats. The acceleration column supplements that catalogue with
matching-assistant information and the documented DFlash2 target constraints.
Model loading constraints live in `docs/troubleshooting.md`; weight labels and
download operations live in the user guide. Runtime limitations live in the
corresponding API and acceleration guides.
