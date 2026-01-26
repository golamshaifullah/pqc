# Docs Generation Guide

This repository uses Google-style docstrings and type hints to generate API
documentation directly from the source.

## Recommended Toolchain

- Sphinx with:
  - `sphinx.ext.autodoc`
  - `sphinx.ext.autosummary`
  - `sphinx.ext.napoleon` (Google-style docstrings)
  - `sphinx.ext.intersphinx` (optional)

MkDocs with `mkdocstrings[python]` can also work, but the steps below assume
Sphinx because it is the most common for Python API references.

## Build Steps (Sphinx)

1. Create a docs folder and initialize Sphinx:

   ```bash
   mkdir -p docs
   cd docs
   sphinx-quickstart
   ```

2. Update `docs/conf.py` to include:

   ```python
   import os
   import sys
   sys.path.insert(0, os.path.abspath(".."))

   extensions = [
       "sphinx.ext.autodoc",
       "sphinx.ext.autosummary",
       "sphinx.ext.napoleon",
       "sphinx.ext.intersphinx",
   ]

   autosummary_generate = True
   napoleon_google_docstring = True
   napoleon_numpy_docstring = False
   ```

3. Create an API index file (e.g., `docs/api.rst`):

   ```rst
   API Reference
   =============

   .. autosummary::
      :toctree: generated
      :recursive:

      pqc
      pqc.cli
      pqc.config
      pqc.pipeline
      pqc.detect
      pqc.detect.bad_measurements
      pqc.detect.ou
      pqc.detect.transients
      pqc.features
      pqc.features.backend_keys
      pqc.io
      pqc.io.libstempo_loader
      pqc.io.merge
      pqc.io.timfile
      pqc.utils
      pqc.utils.diagnostics
      pqc.utils.logging
      pqc.utils.stats
   ```

4. Build the docs:

   ```bash
   make html
   ```

5. Open `docs/_build/html/index.html` locally to view the manual.

## Hosting

You can serve the built HTML locally with:

```bash
python -m http.server --directory docs/_build/html
```

## Notes

- Ensure `pqc` is installed (editable or otherwise) so autodoc can import it.
- If libstempo is not installed in the docs environment, you may need to mock
  its imports via `autodoc_mock_imports = ["libstempo"]` in `conf.py`.
