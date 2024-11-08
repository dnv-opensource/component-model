# ruff: noqa
# mypy: ignore-errors

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../src").absolute()))
sys.path.insert(0, str(Path("../_ext").absolute()))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "component-model"
copyright = "2024, DNV AS. All rights reserved."
author = "Siegfried Eisinger"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_argparse_cli",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
    "get_from_code",
    "spec",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    # "sphinx.ext.autosectionlabel",
]
todo_include_todos = True
spec_include_specs = True

# The file extensions of source files.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = f"component-model {release}"
html_theme = "furo"
# html_theme = "classic"  # alabaster'
# html_theme_options = {
#     "rightsidebar": "false",
#     "stickysidebar": "true",
#     "relbarbgcolor": "black",
#     "body_min_width": "700px",
#     "body_max_width": "900px",
#     "sidebarwidth": "250px",
# }
html_static_path = ["_static"]
html_logo = "_static/component-model.svg"
autodoc_default_options = {
    "member-order": "groupwise",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_preserve_defaults = True

myst_heading_anchors = 3

# add markdown mermaid support
myst_fence_as_directive = ["mermaid"]
