# Contributing

Thanks for helping. Bug reports, fixes, new backends and documentation are all welcome; open an
[issue](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/issues) first for anything large.

## Development setup

```bash
git clone https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git
cd WholebodyPoseEstimation
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU PyTorch for ViTPose
pip install -e ".[all,dev]"
pre-commit install
```

`pip install -e ".[dev]"` without backends is enough for changes to the core (API, I/O, video
pipeline, filters, drawing, CLI): the unit tests use a fake backend.

## Tests

```bash
pytest -m "not slow"     # unit tests: no backend, no download, a few seconds
pytest -m slow           # real backends on data/images/image.jpg (downloads weights on first run)
pytest                   # everything
```

- Unit tests must not need a backend or the network. Use `tests/conftest.py::FakePoseModel` and the
  `synthetic_video` fixture; guard anything that needs an optional package with
  `pytest.importorskip("<module>")`.
- Tests that run a real model are marked `@pytest.mark.slow` (or the whole module with
  `pytestmark = pytest.mark.slow`).
- CI runs the unit tests on Python 3.10-3.13 with only the core installed, and the slow tests with
  every backend on `main`, weekly and on demand ([`ci.yml`](.github/workflows/ci.yml)).

## Style

- [ruff](https://docs.astral.sh/ruff/) for linting and formatting, configured in `pyproject.toml`
  (line length 120). `pre-commit` runs it on every commit; by hand:
  `ruff check --fix . && ruff format .` (this also formats Python code blocks in Markdown).
- Type hints on public functions, Google-style docstrings with shapes, e.g. `keypoints (T, 133, 2)`.
- Images are **RGB** at the API boundary; convert to BGR inside a backend if its model needs it.
- Keep `import wholebodypose` dependency-free beyond NumPy and OpenCV: import backend packages inside
  the backend module, and raise an `ImportError` that names the extra to install.
- Update `CHANGELOG.md` (section *Unreleased*) and the relevant page in `docs/` with user-visible changes.

## Adding a backend

1. Create `src/wholebodypose/models/<name>/model.py` with a subclass of
   `wholebodypose.models.base.BasePoseModel`:

   ```python
   import numpy as np

   from wholebodypose.models.base import BasePoseModel


   class MyModel(BasePoseModel):
       name = "mymodel"

       def __init__(self, device="auto", use_thresholding=False, kpt_thr=0.5, verbose=True):
           super().__init__(use_thresholding=use_thresholding, kpt_thr=kpt_thr)
           try:
               import mypackage  # noqa: F401
           except ImportError as e:
               raise ImportError("MyModel backend needs: pip install 'wholebodypose[mymodel]'") from e
           ...

       def _predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
           # return (N, 133, 2) pixel keypoints and (N, 133) scores in COCO-WholeBody order,
           # N = 0 (empty arrays) when nobody is found
           ...
   ```

   `BasePoseModel.predict` reshapes the output, applies `use_thresholding` / `kpt_thr`, and is what
   users call. Undetected keypoints must be `(0, 0)` with score `0`.
2. Download weights through `wholebodypose.weights.get_weight`: add a `WeightFile` entry to `WEIGHTS`
   (mirror path, upstream URL, SHA-256 if known, license).
3. Register it in `wholebodypose.api.BACKENDS`:

   ```python
   Backend(
       "mymodel",
       "wholebodypose.models.mymodel.model:MyModel",
       "mymodel",  # pip extra
       "Short description shown by `wholebodypose backends`",
       {"use_thresholding": True, "kpt_thr": 0.5},  # defaults applied by PoseEstimator
       multiperson=False,
   )
   ```

   `create_model` passes `verbose` to every backend and `device` to every backend except `mediapipe`.
4. Add the extra to `[project.optional-dependencies]` in `pyproject.toml` (and to `all` if it installs
   cleanly everywhere).
5. Add a case to `tests/test_backends_slow.py`, a row to the tables in the README and in
   `docs/guide/backends.md`, the citation, and the weight license to `THIRD_PARTY.md`.

`tests/test_api.py::test_registry_lists_paper_backends` pins the list of backends; update it too.

## Pull requests

- One topic per pull request, with tests for bug fixes and new features.
- `pre-commit run --all-files` and `pytest -m "not slow"` pass locally.
- Describe how you tested it (backend, device, inputs) in the PR template.
- Contributions are accepted under the project license (Apache-2.0, see
  [docs/LICENSING.md](docs/LICENSING.md)). Code taken from another project keeps its license: say so in the file
  header and in [THIRD_PARTY.md](THIRD_PARTY.md).

## Releases (maintainers)

1. Move the *Unreleased* entries of `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD` and set
   `version = "X.Y.Z"` in `pyproject.toml`, and `version` / `date-released` in `CITATION.cff`.
2. `git tag -a vX.Y.Z -m "WholeBodyPose X.Y.Z" && git push origin vX.Y.Z`.
3. [`release.yml`](.github/workflows/release.yml) checks that the tag matches the version, builds the
   sdist and wheel and publishes a GitHub Release with that CHANGELOG section.
4. PyPI (TestPyPI first) is a separate, manual run of the same workflow; see
   [docs/release.md](docs/release.md) for the one-time Trusted Publishing setup and the steps.
