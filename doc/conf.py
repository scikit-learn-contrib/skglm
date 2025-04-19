# -*- coding: utf-8 -*-
#
# picard documentation build configuration file, created by
# sphinx-quickstart on Mon May 23 16:22:52 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys
import warnings
from datetime import date

import sphinx_gallery  # noqa
from numpydoc import numpydoc, docscrape  # noqa
import pydata_sphinx_theme  # noqa

from skglm import __version__ as version

# include custom extension
curdir = os.path.dirname(__file__)  # noqa
sys.path.append(os.path.abspath(os.path.join(curdir, 'sphinxext')))  # noqa

from github_link import make_linkcode_resolve


# Mathurin: disable agg warnings in doc
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

warnings.filterwarnings("ignore", category=UserWarning,
                        message="Trying to register the cmap")

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'sphinx_design',
    'numpydoc',
    'sphinx.ext.linkcode',
    'gh_substitutions',
    'myst_parser',
    'sphinx_sitemap',
    # custom ext, see ./sphinxext/gh_substitutions.py
]

# set it to True to build a stable version of the documentation
is_stable_doc = False

myst_enable_extensions = [
    "dollarmath",
    "amsmath"
]
# generate autosummary even if no references
autosummary_generate = True

templates_path = ['_templates']

html_css_files = ["style.css"]

source_suffix = '.rst'
master_doc = 'index'

project = u'skglm'
copyright = f'2022-{date.today().year}, skglm developers'
author = u'skglm developers'

release = version

language = 'en'

exclude_patterns = ['_build']

# pygments_style = 'sphinx'

todo_include_todos = False

html_baseurl = 'https://contrib.scikit-learn.org/skglm/'

extensions.append("sphinxext.opengraph")

# OpenGraph config
ogp_site_url = html_baseurl
ogp_image = "https://contrib.scikit-learn.org/skglm/_static/images/logo.svg"
ogp_description_length = 250
ogp_type = "website"


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

dev_color = "#ff7f0e"
stable_color = "#1f77b4"

primary = stable_color if is_stable_doc else dev_color
primary_dark = "#135b91" if is_stable_doc else "#e76f00"

version_match = "stable" if is_stable_doc else "dev"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # color configurations
    "light_css_variables": {
        "color-brand-primary": primary,
        "color-brand-content": primary,
        "color-brand-secondary": primary_dark,
    },
    "dark_css_variables": {
        "color-brand-primary": primary_dark,
        "color-brand-content": primary_dark,
        "color-brand-secondary": primary,
    },

    # Navbar configuration
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "navbar-icon-links",
        "theme-switcher",
        "version-switcher",
    ],

    # other configurations
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/scikit-learn-contrib/skglm",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
    "version_dropdown": True,
    "switcher": {
        # NOTE: must be a URL and not a path relative to doc
        "json_url": "https://raw.githubusercontent.com/floriankozikowski/skglm/refs/heads/documentation-update/doc/_static/switcher.json",
        "version_match": version_match,
    },
    "navigation_with_keys": True,
    "search_bar_text": "Search skglm docs...",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/scikit-learn-contrib/skglm",
            "icon": "fa-brands fa-github",
        },
    ],
}

# add sidebars if necessary (e.g. if more tutorials are being added)
html_sidebars = {
    "index": [],
    "getting_started": [],
    "tutorials": [],
    "tutorials/*": [],
    "auto_examples/*": [],
    "api": [],  # not applied for subpages as it is useful here
    "contribute": [],
    "changes": [],
    "changes/*": []
}

# Enable asciimath parsing in MathJax and configure the HTML renderer to output
# the default asciimath delimiters. Asciimath will not be correctly rendered in
# other output formats, but can likely be fixed using py-asciimath[1] to convert
# to Latex.
# [1]: https://pypi.org/project/py-asciimath/
mathjax3_config = {
    "loader": {
        "load": ['input/asciimath']
    },
}
mathjax_inline = ['`', '`']
mathjax_display = ['`', '`']

html_static_path = ['_static']
html_js_files = [
    "scripts/asciimath-defines.js",
    "switcher.json"
]

# -- Options for copybutton ---------------------------------------------
# complete explanation of the regex expression can be found here
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#using-regexp-prompt-identifiers
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f"skglm {version} documentation"

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/images/logo.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/images/logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
html_extra_path = ['robots.txt']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# these variables will be available in the sphinx templates
html_context = {
    "is_stable_doc": is_stable_doc
}


# when it's the dev version of the documentation, put a banner to warn the user
# and a link to switch to the dev version of doc
if not is_stable_doc:
    html_theme_options["announcement"] = (
        "You are viewing the documentation of the dev version of "
        "<code>skglm</code> which contains WIP features. "
        "View <a href='https://contrib.scikit-learn.org/skglm/stable/index.html'>"
        "stable version</a>."
    )


# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr'
# html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
# html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
# html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'skglmdoc'

numpydoc_show_class_members = False

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',

    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'skglm.tex', u'skglm Documentation',
     u'skglm developers', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'skglm', u'skglm Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'skglm', u'skglm Documentation',
     author, 'skglm', 'One line description of project.',
     'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    # 'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    # 'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    # 'matplotlib': ('https://matplotlib.org/', None),
    'benchopt': ('https://benchopt.github.io', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
}

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "skglm",
    "https://github.com/scikit-learn-contrib/"
    "skglm/blob/{revision}/"
    "{package}/{path}#L{lineno}",
)

sphinx_gallery_conf = {
    # 'backreferences_dir': 'gen_modules/backreferences',
    'backreferences_dir': 'generated',
    'doc_module': ('skglm', 'sklearn', 'benchopt'),
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'reference_url': {
        'skglm': None,
    }
}
