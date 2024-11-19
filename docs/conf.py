# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ATCG Lib"
copyright = "2024, Domenic Zingsheim"
author = "Domenic Zingsheim"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["breathe", "myst_parser"]

source_suffix = [".rst", ".md"]

# breathe_projects = {
#     "ATCG Lib": "../bin/doxygen/xml/",
# }

breathe_default_project = "ATCGLIB"
breathe_domain_by_extension = {"h": "cpp", "cpp": "cpp"}
cpp_index_common_prefix = [
    "atcg::",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
# html_static_path = ["_static"]
html_theme_options = {
    "toc_title": "Navigation",  # Customize the sidebar title
    "show_toc_level": 1,  # Control depth of sidebar links
    "collapse_navbar": False,  # Ensure the sidebar doesn't collapse subitems
}
