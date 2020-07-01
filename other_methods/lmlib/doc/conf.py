# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
from sphinx_gallery.sorting import ExplicitOrder

sys.path.insert(0, os.path.abspath("../"))

from doc.var_subs import var_sub

rst_epilog = var_sub
# -- Project information -----------------------------------------------------

project = "lmlib"
copyright = "2019, ETH, BFH, Authors: Frédéric Waldmann, Reto Wildhaber"
author = "Frédéric Waldmann, Reto Wildhaber"

# The full version, including alpha/beta/rc tags
release = "0.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_gallery.gen_gallery",
]

source_suffix = ".rst"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# generate autosummary even if no references
autosummary_generate = True
autoclass_content = "both"
autodoc_default_options = {"show-inheritance": True}
autodoc_member_order = "bysource"
add_module_names = False

# Napoleon conf
napoleon_use_rtype = True

# ------ sphinx gallery config ----------------------------------------------------------------------------
sphinx_gallery_conf = {
    # just example gallery
    # 'examples_dirs': '../examples',   # path to your example scripts
    # 'gallery_dirs': '_autoexamples', # path where to save gallery generated output
    # both galleries example and tutorials
    "examples_dirs": ["../examples", "../tutorials"],  # path to your example scripts
    "subsection_order": ExplicitOrder(["../examples/basic", "../examples/advanced"]),
    "gallery_dirs": [
        "_autoexamples",
        "_autotutorials",
    ],  # path where to save gallery generated output
}

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
highlight_language = "python3"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_logo = "_static/logo.svg"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "mayavi": ("http://docs.enthought.com/mayavi/mayavi", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "sphinx": ("http://www.sphinx-doc.org/en/stable", None),
    "pandas": ("https://pandas.pydata.org/", None),
}

# --------Options for LATEX output


# latex_toplevel_sectioning = 'part'
latex_logo = "_static/logo.svg"

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    "preamble": "\\addto\\captionsenglish{\\renewcommand{\\contentsname}{Table of contents}}",
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
