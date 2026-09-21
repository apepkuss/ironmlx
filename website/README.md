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
