"""Sphinx configuration for mktlib documentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to sys.path so autodoc can find the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "mktlib"
author = "Matt Buck"
copyright = "2025, Matt Buck"  # noqa: A001

# The short X.Y version
version = "0.5"
# The full version, including alpha/beta/rc tags
release = "0.5.4"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_sitemap",
    "sphinxext.opengraph",
]

language = "en"

templates_path = ["_templates"]
exclude_patterns = ["_build", "CODEMAPS"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_baseurl = "https://polars-mktlib.readthedocs.io/en/latest/"
sitemap_url_scheme = "{link}"

# -- autodoc settings --------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
always_use_bars_union = True
autodoc_mock_imports = ["polars_sdist", "polars_rfft"]

# -- intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

# -- Open Graph settings -----------------------------------------------------

ogp_site_url = "https://polars-mktlib.readthedocs.io/en/latest/"
ogp_site_name = "mktlib"
ogp_description_length = 200
