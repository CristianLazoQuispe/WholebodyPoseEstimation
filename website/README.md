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
checks that every local `src`/`href`/`poster` in `index.html` exists. One-time repository setting:
**Settings → Pages → Source: GitHub Actions**.

## Layout

| Path | What it is |
|---|---|
| `index.html` | the page: header and links, teaser video, TL;DR and abstract (from the paper), key contributions, pipeline (inline SVG, with a vertical variant below 680 px), keypoint layouts (paper Figure 1 a/b), backends, results (paper Table 3), quick start, pose datasets, demo (link + click-to-load iframe of the Hugging Face Space), paper and BibTeX, footer |
| `static/css/style.css` | the page style |
| `static/js/main.js` | teaser play/pause (paused under `prefers-reduced-motion`), copy buttons, click-to-load demo iframe |
| `static/videos/teaser.{webm,mp4}` | the teaser (`docs/assets/output_telefono_pucp_slow.gif`, 5.6 MB) re-encoded to VP9 / H.264 (~0.45 MB each) |
| `static/images/teaser_poster.jpg` | poster frame of the teaser (also the `<img>` fallback) |
| `static/images/og.png` | 1200×630 social card (`og:image`, `twitter:image`) |
| `static/images/favicon.svg`, `favicon-32.png`, `apple-touch-icon.png` | icons |
| `static/images/kpts_*.jpg`, `paper_thumb.jpg` | Figure 1 (a) and (b) extracted from the paper PDF, and its first page |
| `static/paper/WholeBodyPose_ICMV2025.pdf` | the paper |
| `.nojekyll` | serve the folder as-is (no Jekyll processing) |

The text in `index.html` mirrors the repository README (TL;DR, backends, datasets, BibTeX) and the paper
(abstract, contributions, Table 3). When one of them changes, update the page too; the BibTeX must stay
identical to the README's.

## Regenerating the media

All generated files are committed; regenerate them with

```bash
python scripts/make_website_assets.py            # teaser webm/mp4 + poster, og.png, favicons
python scripts/make_website_assets.py --paper    # + paper thumbnail and Figure 1 crops
python scripts/make_website_assets.py --skip-video   # only og.png and favicons
```

It needs Pillow and `ffmpeg` (`--paper` also needs poppler: `pdftoppm`, `pdfimages`). To use another
teaser clip, replace `docs/assets/output_telefono_pucp_slow.gif` (or change `TEASER_GIF` / `POSTER_TIME` in the
script) and rerun it; the OG card is built from the poster frame.

## Old project page

The previous page lived in the `CristianLazoQuispe/wholebodypose.github.io` repository.
`docs/legacy-site-redirect/index.html` is a redirect (meta refresh + canonical + JS) to this page, meant to be
pushed to that repository (as `index.html`, and optionally `404.html`). It is kept outside `website/` so it is
not deployed here.
