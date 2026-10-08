"""Regression checks for generated documentation and Markdown navigation."""
import importlib.util
from html.parser import HTMLParser
from pathlib import Path
import json
import subprocess
import sys
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("website_docs", ROOT / "scripts/build-website-docs.py")
DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCS)


class PageParser(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = []
        self.references = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for name in ("href", "src"):
            if name in attrs:
                self.references.append(attrs[name])


class ModelTableParser(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.rows = []
        self.row = None
        self.value = None
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.value = ""

    def handle_data(self, value):
        if self.value is not None:
            self.value += value

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.value is not None:
            self.row.append(self.value.strip())
            self.value = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


class WebsiteDocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable, str(ROOT / "scripts/build-website-docs.py")], check=True, capture_output=True)
        cls.website = ROOT / "website"
        cls.pages = {}
        for directory in (cls.website / "docs", cls.website / "zh-Hans/docs"):
            for path in directory.rglob("*.html"):
                cls.pages[path.resolve()] = PageParser(path.read_text(encoding="utf-8"))

    def test_published_pages_have_unique_ids_and_resolving_references(self):
        self.assertEqual(len(self.pages), 2 * (len(DOCS.PAGES) + 1))
        for path, page in self.pages.items():
            with self.subTest(page=path.relative_to(self.website)):
                self.assertEqual(len(page.ids), len(set(page.ids)))
                for reference in page.references:
                    url = urlsplit(reference)
                    if url.scheme or url.netloc:
                        continue
                    target = (path.parent / unquote(url.path)).resolve() if url.path else path
                    if target.is_dir():
                        target /= "index.html"
                    self.assertTrue(target.exists(), f"{path}: {reference}")
                    if url.fragment and target in self.pages:
                        self.assertIn(unquote(url.fragment), self.pages[target].ids, reference)

    def test_bilingual_search_links_include_real_section_anchors(self):
        for directory in (self.website / "docs", self.website / "zh-Hans/docs"):
            data = json.loads((directory / "search-index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data), len(DOCS.PAGES))
            self.assertEqual(len({doc["href"] for doc in data}), len(DOCS.PAGES))
            for doc in data:
                target = (directory / doc["href"]).resolve()
                self.assertIn(target, self.pages)
                for section in doc["sections"]:
                    self.assertIn(section["anchor"], self.pages[target].ids)

    def test_dsh_is_published_with_cli_desktop_and_local_integration_links(self):
        for directory in (self.website / "docs", self.website / "zh-Hans/docs"):
            with self.subTest(language=directory):
                target = directory / "dsh.html"
                html = target.read_text(encoding="utf-8")
                self.assertIn("DSH CLI", html)
                self.assertIn("DSH Desktop", html)
                self.assertIn('<th>', html)
                self.assertIn('<td><code>ironmlx-local</code></td>', html)
                self.assertIn('<ol start="4">', html)
                self.assertIn('href="dsh.html"', (directory / "index.html").read_text(encoding="utf-8"))
                for name in ("user-guide.html", "api-reference.html"):
                    article = (directory / name).read_text(encoding="utf-8")
                    self.assertIn('href="dsh.html"', article)
                    self.assertNotIn('blob/dev/docs/dsh.md', article)
                    self.assertNotIn('blob/dev/docs/zh-CN/dsh.md', article)
                data = json.loads((directory / "search-index.json").read_text(encoding="utf-8"))
                dsh = next(doc for doc in data if doc["href"] == "dsh.html")
                self.assertTrue(any("DSH Desktop" in section["title"] for section in dsh["sections"]))
                self.assertTrue(any("DSH CLI" in section["title"] for section in dsh["sections"]))
        self.assertEqual(DOCS.GROUP_MAP["dsh.md"][0], "agents")
        self.assertEqual(DOCS.GROUP_MAP["dflash2-server-api.md"][0], "inference")
        self.assertEqual(DOCS.GROUP_MAP["automatic-updates.md"][0], "maintenance")
        self.assertEqual(DOCS.GROUP_MAP["building-from-source.md"][0], "api")
        self.assertEqual(DOCS.GROUP_MAP["security-boundary.md"][0], "security")

    def test_model_lists_cover_the_app_catalogue_without_claiming_unverified_runs(self):
        catalog = json.loads((ROOT / "ironmlx-app/Sources/IronMLXAppCore/Resources/supported-models.json").read_text(encoding="utf-8"))
        groups = {}
        for entry in catalog["entries"]:
            groups.setdefault(entry["modelId"], []).append(entry)
        expected_repositories = {"https://huggingface.co/" + entry["hfRepo"] for entry in catalog["entries"]}
        for directory in (self.website / "docs", self.website / "zh-Hans/docs"):
            chinese = directory.name == "docs" and directory.parent.name == "zh-Hans"
            html = (directory / "supported-models.html").read_text(encoding="utf-8")
            rows = ModelTableParser(html).rows
            for row in rows:
                self.assertNotIn("验证" if chinese else "Validation", row)
            self.assertEqual(rows[0][-1], "加速类型" if chinese else "Acceleration type")
            model_rows = {row[0]: row for row in rows if row[0] in {entries[0]["name"] for entries in groups.values()}}
            self.assertEqual(len(model_rows), len(groups))
            repositories = {reference for reference in PageParser(html).references if reference.startswith("https://huggingface.co/")}
            self.assertEqual(repositories, expected_repositories)
            for entries in groups.values():
                row = model_rows[entries[0]["name"]]
                if entries[0]["category"] in {"text", "vision"}:
                    self.assertEqual(len(row), 8)
                    continue
                self.assertEqual(len(row), 4)
            self.assertEqual(model_rows["DiffusionGemma 26B A4B IT"][7], "—")
            self.assertEqual(model_rows["Qwen 3.8 27B"][7], "MTP / DFlash2")
            self.assertEqual(model_rows["Gemma 4 E4B IT"][7], "Assistant")
            self.assertEqual(model_rows["Qwen 3.5 2B"][7], "—")
            self.assertEqual(model_rows["Qwen 3.5 4B"][7], "MTP")
            self.assertEqual(model_rows["Llama 3.2 1B Instruct"][4:7], ["—", "—", "✓"])
            self.assertNotIn("Generated by", html)
            self.assertIn('href="troubleshooting.html#', html)
            for name in ("audio-speech-api", "image-generation-api", "laya-systemone-api"):
                self.assertIn(f'href="{name}.html"', html)
                self.assertIn((directory / f"{name}.html").resolve(), self.pages)
                self.assertFalse(any(name + ".md" in reference and "/blob/" in reference for reference in PageParser(html).references))

        subprocess.run([sys.executable, str(ROOT / "scripts/build-supported-models-docs.py"), "--check"], check=True, capture_output=True)

    def test_model_generator_does_not_extend_support_to_new_variants(self):
        spec = importlib.util.spec_from_file_location("model_docs", ROOT / "scripts/build-supported-models-docs.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        entries = [
            {"name": "Qwen 3.8 27B", "status": "verified", "weightFormat": "Affine 4-bit", "capabilities": ["vision"]},
            {"name": "Qwen 3.8 27B", "status": "compatible", "weightFormat": "BF16", "capabilities": []},
        ]
        self.assertEqual(generator.capability(entries, "vision", "en"), "By variant")
        self.assertEqual(generator.acceleration(entries, set()), "—")

    def test_embeddings_are_published_under_the_api_hub_with_catalogue_variants(self):
        self.assertEqual(DOCS.GROUP_MAP["text-embeddings.md"][0], "api")
        self.assertEqual(DOCS.PAGE_PARENT["text-embeddings.md"], "api-reference.md")
        for directory in (self.website / "docs", self.website / "zh-Hans/docs"):
            with self.subTest(language=directory):
                topic = (directory / "text-embeddings.html").read_text(encoding="utf-8")
                self.assertIn('href="api-reference.html"', topic)
                self.assertIn('href="service-api.html"', topic)
                self.assertIn('<code>POST /v1/embeddings</code>', topic)
                hub = (directory / "api-reference.html").read_text(encoding="utf-8")
                self.assertIn('href="text-embeddings.html#', hub)
                self.assertIn('<code>/v1/embeddings</code>', hub)
                model_html = (directory / "supported-models.html").read_text(encoding="utf-8")
                row = next(row for row in ModelTableParser(model_html).rows if row[0] == "EmbeddingGemma 2")
                self.assertEqual(row[1], "Embedding")
                self.assertIn("BF16", row[3])
                self.assertIn("Affine 4-bit", row[3])
                self.assertIn('href="text-embeddings.html"', model_html)
                for retired in ("api", "api-compatibility-matrix", "tts-model-download", "laya-phase-one-contract"):
                    self.assertFalse((directory / f"{retired}.html").exists())

    def test_repository_documents_have_bilingual_web_destinations(self):
        links = {
            "audio-speech-api": "audio-library",
            "diagnostic-bundle": "support",
            "troubleshooting": "support",
            "user-guide": "contributing",
            "support": "security",
            "contributing": "security",
        }
        for language, directory, counterpart in (
            ("en", self.website / "docs", "../zh-Hans/docs/"),
            ("zh", self.website / "zh-Hans/docs", "../../docs/"),
        ):
            for source, destination in links.items():
                html = (directory / f"{source}.html").read_text(encoding="utf-8")
                article = html.split('<article class="doc-content">', 1)[1].split('</article>', 1)[0]
                self.assertIn(f'href="{destination}.html"', article)
            for name in ("audio-library", "support", "contributing", "security", "laya-systemone-api"):
                html = (directory / f"{name}.html").read_text(encoding="utf-8")
                self.assertIn(f'href="{counterpart}{name}.html"', html)
                canonical = DOCS.source_path(name + ".md", language).relative_to(ROOT).as_posix()
                self.assertIn(f'https://github.com/apepkuss/ironmlx/edit/dev/{canonical}', html)
            audio = (directory / "audio-library.html").read_text(encoding="utf-8")
            self.assertIn('class="doc-diagram"', audio)
            self.assertNotIn('class="language-mermaid"', audio)
            self.assertIn('>复制</button>' if language == "zh" else '>Copy</button>', audio)

    def test_renderer_preserves_code_and_multiline_instructions(self):
        source = ROOT / "docs/zh-CN/developer-guide.md"
        target = self.website / "zh-Hans/docs/developer-guide.html"
        markdown = '# Demo\n\n1. Install the app\n   and open it.\n2. Load a model.\n\n```json\n{"input":"<script>& test"}\n```\n\n## Repeated heading\n\n## Repeated heading\n'
        content, toc, _ = DOCS.render(markdown, source, target)
        self.assertIn("<li>Install the app and open it.</li>", content)
        self.assertIn('{&quot;input&quot;:&quot;&lt;script&gt;&amp; test&quot;}', content)
        self.assertNotIn("<script>", content)
        self.assertEqual([item[2] for item in toc], ["repeated-heading", "repeated-heading-1"])
        self.assertIn('id="section-2"', content)

    def test_nested_release_pages_keep_overview_and_language_destinations(self):
        target = self.website / "zh-Hans/docs/release-notes/0.2.0.html"
        html = target.read_text(encoding="utf-8")
        self.assertIn('href="../index.html"', html)
        self.assertIn('href="../../../docs/release-notes/0.2.0.html"', html)
        self.assertIn('data-search-index="../search-index.json"', html)

    def test_inline_code_and_external_resource_links_are_safe(self):
        source = ROOT / "docs/zh-CN/supported-models.md"
        target = self.website / "zh-Hans/docs/supported-models.html"
        content = DOCS.inline('`[x](developer-guide.md)` [API](text-vision-api.md#openai-responses-api) [JSON](../../ironmlx-app/Sources/IronMLXAppCore/Resources/supported-models.json)', source, target)
        self.assertIn('<code>[x](developer-guide.md)</code>', content)
        self.assertIn('href="text-vision-api.html#openai-responses-api"', content)
        self.assertIn('https://github.com/apepkuss/ironmlx/blob/dev/ironmlx-app/Sources/IronMLXAppCore/Resources/supported-models.json', content)
        upstream = "https://github.com/Blaizzy/mlx-vlm/blob/main/README.md#usage"
        self.assertIn(f'href="{upstream}"', DOCS.inline(f'[Upstream]({upstream})', source, target))


if __name__ == "__main__":
    unittest.main()
