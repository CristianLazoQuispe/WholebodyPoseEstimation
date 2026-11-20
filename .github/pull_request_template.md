## What and why

<!-- What does this change and why? Link the issue: "Fixes #123". -->

## How it was tested

<!-- Commands you ran; for model changes: backend, device and inputs, before/after. -->

- [ ] `pytest -m "not slow"`
- [ ] `pytest -m slow` (needed when a backend, the weights or the video pipeline changes)
- [ ] `pre-commit run --all-files`

## Checklist

- [ ] Tests added or updated
- [ ] Docs (`docs/`, README) updated for user-visible changes
- [ ] `CHANGELOG.md` entry under *Unreleased*
- [ ] Paper defaults (`wholebodypose.api.BACKENDS`) unchanged, or the change is explained above
