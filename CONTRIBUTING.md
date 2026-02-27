# Contributing to PQC

Thanks for contributing.

## Development setup

```bash
git clone https://github.com/golamshaifullah/pqc.git
cd pqc
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you need to run the full pipeline locally:

```bash
python -m pip install -e ".[libstempo]"
```

## Local checks before opening a PR

```bash
python -m ruff check .
python -m black --check .
python -m pytest
```

Build docs:

```bash
make -C docs html
```

## Pull request guidelines

- Keep PRs focused (one logical change).
- Add/update tests for behavioral changes.
- Update docs for user-facing changes (README/docs and API docstrings).
- If config/CLI semantics change, include migration notes in the PR description.
- Ensure CI is green.

Use the PR templates in `.github/PULL_REQUEST_TEMPLATE/` for feature and bugfix PRs.

## Commit message style (recommended)

- `feat: ...` for new functionality
- `fix: ...` for bug fixes
- `docs: ...` for docs-only changes
- `refactor: ...` for internal structural changes
- `test: ...` for tests-only changes
