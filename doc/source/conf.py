# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os, sys, shutil


sys.path.insert(0, os.path.abspath("../_ext"))
sys.path.insert(0, os.path.abspath("../../"))
print("PATH", sys.path)
shutil.copyfile("../../README.rst", "readme.rst")


# -- Project information -----------------------------------------------------

project = "Component Model"
copyright = "2023, DNV"
author = "Siegfried Eisinger"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",  # to handle README.md
    "sphinx.ext.todo",
    "get_from_code",
    "spec",
    "sphinx.ext.napoleon",  # to read nupy docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
]
todo_include_todos = True
spec_include_specs = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
autoclass_content = "both"  # both __init__ and class docstring

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "classic"  # alabaster'
html_theme_options = {
    "body_min_width": "70%",
    "rightsidebar": "false",
    "stickysidebar": "true",
    "relbarbgcolor": "black",
    "body_min_width": "700px",
    "body_max_width": "900px",
    "sidebarwidth": "250px",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
