# Project page

Static project page for *WholeBodyPose: A Unified End-to-End Framework for Sign Language Recognition and
Pose-Based Training Data* (ICMV 2025, Proceedings of SPIE), served by GitHub Pages at
<https://cristianlazoquispe.github.io/WholebodyPoseEstimation/>.

Plain HTML/CSS/vanilla JS: no build step, no framework. The only external resource is Google Fonts (the page
falls back to system fonts without it). Every asset path is relative, so the page works under the
`/WholebodyPoseEstimation/` sub-path and from a local server alike. Layout in the usual academic project-page
style (Nerfies): light theme, one centred column.

## Preview

```bash
python -m http.server 8000 -d website
# open http://localhost:8000
```

## Deployment

`.github/workflows/pages.yml` uploads this folder as the Pages artifact and deploys it on every push to
`main` that touches `website/**` or the workflow (and on manual `workflow_dispatch`). Before uploading, it
checks that every local `src`/`href`/`poster` in `index.html` exists, that the schema.org JSON-LD parses
and holds the paper, software and dataset entities, that the Google Scholar `citation_*` tags are present, and
that `sitemap.xml`, `robots.txt` and `llms.txt` exist. One-time repository setting:
**Settings → Pages → Source: GitHub Actions**.

## Layout

| Path | What it is |
|---|---|
| `index.html` | the page: header and links, pipeline figure (inline SVG, with a vertical variant below 680 px), TL;DR and abstract (from the paper), key contributions, backends and keypoint layouts (paper Figure 1 a/b), datasets (one row per published subset with links to every split file and label map on the Hub), data processing, ISLR models (paper Table 3 with links to each model's original code), quick start, demo (link + click-to-load iframe of the Hugging Face Space), paper and BibTeX, acknowledgements |
| `static/css/style.css` | the page style |
| `static/js/main.js` | copy buttons, click-to-load demo iframe |
| `static/images/og.png` | 1200×630 social card (`og:image`, `twitter:image`), built from the Figure 1 crops |
| `static/images/favicon.svg`, `favicon-32.png`, `apple-touch-icon.png` | icons |
| `static/images/kpts_*.jpg`, `paper_thumb.jpg` | Figure 1 (a) and (b) extracted from the paper PDF, and its first page |
| `static/paper/WholeBodyPose_ICMV2025.pdf` | the paper |
| `.nojekyll` | serve the folder as-is (no Jekyll processing) |
| `llms.txt` | short Markdown map of the project for LLM tools (docs, datasets, citation, license, install) |
| `sitemap.xml`, `robots.txt` | sitemap for search engines; `robots.txt` allows everything but only works at the host root |

`<head>` also carries the Google Scholar `citation_*` tags and a schema.org JSON-LD block
(`ScholarlyArticle`, `SoftwareSourceCode`, `Dataset`); keep them in sync with the BibTeX and the dataset card.

The text in `index.html` mirrors the repository README (TL;DR, backends, BibTeX), the paper (abstract,
contributions, Table 3), the dataset card `hf/dataset_cards/pose-action-recognition.md` (subset counts, sizes,
license) and `docs/datasets/` (data processing). When one of them changes, update the page too; the BibTeX
must stay identical to the README's. When a new subset is published on the Hub (e.g. DGI_PUCP38), add its row
to the datasets table and drop the "coming soon" note from the table caption.

## Regenerating the media

All generated files are committed; regenerate them with

```bash
python scripts/make_website_assets.py            # og.png and favicons
python scripts/make_website_assets.py --paper    # + paper thumbnail and Figure 1 crops (og.png uses them)
```

It needs Pillow (`--paper` also needs poppler: `pdftoppm`, `pdfimages`).
