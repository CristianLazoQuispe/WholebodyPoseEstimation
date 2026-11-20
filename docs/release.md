# Releasing (maintainers)

Releases are cut from a version tag by [`.github/workflows/release.yml`](../.github/workflows/release.yml):
it checks that the tag matches `version` in `pyproject.toml`, builds the sdist and wheel, runs
`twine check --strict`, rejects direct-URL (git) dependencies in the metadata, smoke-tests the wheel in a
clean virtual environment and publishes a GitHub Release with the matching `CHANGELOG.md` section.

Publishing to PyPI is **separate and manual**: it never runs on a tag push. It uses
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OpenID Connect between GitHub Actions
and PyPI), so no API token is stored anywhere.

## Before the first PyPI release

Do these once the package is ready to be public; nothing here has been done yet.

1. **Weights mirror.** `CristianLazoQuispe/wholebodypose-weights` is public. Check a clean download from a machine
   without a Hugging Face login:
   `WHOLEBODYPOSE_HOME=$(mktemp -d) wholebodypose download vitpose_small yolox_tiny rtmw_l_256`.
2. **License.** `LICENSE`, `NOTICE`, the `license` field of `pyproject.toml` (`Apache-2.0 AND MIT`),
   the README badge and `CITATION.cff` agree.
3. **Name.** `wholebodypose` was not taken on PyPI or TestPyPI when this page was written (the JSON API
   returned 404); check again just before registering the publisher.
4. **Pending publishers.** Register the project on both indexes before it exists, as a *pending
   publisher* (pypi.org → Account settings → Publishing → Add a new pending publisher; same on
   test.pypi.org):

   | Field | PyPI | TestPyPI |
   |:--|:--|:--|
   | PyPI project name | `wholebodypose` | `wholebodypose` |
   | Owner | `CristianLazoQuispe` | `CristianLazoQuispe` |
   | Repository name | `WholebodyPoseEstimation` | `WholebodyPoseEstimation` |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `pypi` | `testpypi` |

   TestPyPI is a separate site with its own account.
5. **GitHub environments.** Repository → Settings → Environments: create `testpypi` and `pypi`. For
   `pypi`, add yourself as a *required reviewer* and restrict deployments to tags matching `v*`, so every
   upload to PyPI waits for an explicit approval.
6. **Turn publishing on.** Repository → Settings → Secrets and variables → Actions → Variables: add
   `PYPI_PUBLISHING_ENABLED` = `true`. Without it both publish jobs are skipped.

## Every release

1. Move the *Unreleased* entries of `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD`, set `version = "X.Y.Z"`
   in `pyproject.toml` (and `version` / `date-released` in `CITATION.cff`).
2. Locally: `python -m build && twine check --strict dist/*`, then install the wheel in a fresh virtual
   environment and run `wholebodypose backends` and one image per backend.
3. `git tag -a vX.Y.Z -m "WholeBodyPose X.Y.Z" && git push origin vX.Y.Z` → GitHub Release.
4. Actions → *Release* → *Run workflow*, **use workflow from** the tag `vX.Y.Z`, `publish` = `testpypi`.
   Check the TestPyPI page and install from it in a fresh environment (dependencies come from PyPI):

   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ wholebodypose==X.Y.Z
   ```

5. Run the workflow again from the same tag with `publish` = `pypi`. It uploads to TestPyPI first
   (`skip-existing`), then waits for the `pypi` environment approval and uploads to PyPI. A version can
   be uploaded to PyPI only once; fix mistakes with a new version (`X.Y.Z.post1` or the next patch).

## What PyPI rejects

- Direct-URL dependencies (`name @ git+https://...`) in any extra; CI fails if one appears in the wheel
  metadata.
- A README that does not render (`twine check --strict` catches it).
- Re-uploading a file name that already exists.
