# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ATCG Framework"
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

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo.png"  # Path to your logo
html_favicon = "_static/icon.png"

# Options specific to sphinx_book_theme
html_theme_options = {
    "logo": {
        "text": "ATCG Framework",  # Optional: Text displayed next to the logo
    },
    "home_page_in_toc": True,  # Optional: Show home link in the table of contents
}
html_theme_options = {
    "toc_title": "Navigation",  # Customize the sidebar title
    "show_toc_level": 1,  # Control depth of sidebar links
    "collapse_navbar": False,  # Ensure the sidebar doesn't collapse subitems
}


def preprocess_readme():
    """Rewrite README paths for Sphinx and remove logo."""
    input_path = "../README.md"  # Path to your README file relative to `docs`
    output_path = "processed_README.md"  # Temporary processed file

    with open(input_path, "r") as infile:
        content = infile.read()

    # Rewrite paths from "docs/_static" to "_static"
    idx = content.find("----")
    content = content[(idx + len("----")) :]
    content = "# ATCG Framework Documentation\n" + content

    with open(output_path, "w") as outfile:
        outfile.write(content)

    return output_path


# Preprocess README and use it in the documentation
readme_path = preprocess_readme()
