import os
import sys

# src-layout project: add ../src so autodoc/autosummary can import pqc
sys.path.insert(0, os.path.abspath("../src"))

project = "pqc"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_mock_imports = ["libstempo"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
