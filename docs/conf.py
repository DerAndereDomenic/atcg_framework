# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ATCG Lib'
copyright = '2023, Domenic Zingsheim'
author = 'Domenic Zingsheim'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["breathe"]

# breathe_projects = {
#     "ATCG Lib": "../bin/doxygen/xml/",
# }

breathe_default_project = "ATCGLIB"
breathe_domain_by_extension = {
    "h": "cpp",
    "cpp": "cpp"
}
cpp_index_common_prefix = [
    'atcg::',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
