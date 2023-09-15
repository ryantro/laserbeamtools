# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string
"""
Configuration file for building documentation.

Sphinx builds the docs using couple of external modules: napoleon and nbsphinx.

The overall format is controlled by `.rst` files. The top level file is `index.rst`

`napoleon` builds the API in HTML assuming that the code is documented with
docstrings that follow the Google docstring format.

`nbsphinx` convert the Jupyter notebooks to html with nbsphinx, will
"""

import re
import sys
import os.path

project = 'laserbeamsize'
master_doc = 'index'

def get_init_property(prop):
    """Return property from __init__.py."""
    here = os.path.abspath(os.path.dirname(__file__))
    file_name = os.path.join(here, '..', project, '__init__.py')
    regex = r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop)
    with open(file_name, 'r', encoding='utf-8') as file:
        result = re.search(regex, file.read())
    return result.group(1)

sys.path.insert(0, os.path.abspath('../laserbeamsize'))
release = get_init_property("__version__")
author = get_init_property("__author__")

# -- General configuration ---------------------------------------------------

# Sphinx extension modules
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_automodapi.automodapi',
    'nbsphinx',
]
numpydoc_show_class_members = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_custom_sections = [('Returns', 'params_style')]

# List of patterns, relative to source directory, of files to ignore
exclude_patterns = ['_build',
                    '.tox',
                    '**.ipynb_checkpoints',
                    'readme_images.ipynb',
                    ]

# I execute the notebooks manually in advance.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_scaled_image_link = False
html_sourcelink_suffix = ''
