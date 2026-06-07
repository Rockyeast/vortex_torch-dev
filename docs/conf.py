# -- Path setup --------------------------------------------------------------
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Vortex"
author = "Zhuoming Chen"
copyright = f"{datetime.now():%Y}, {author}"

# Read package version safely (avoid clashing with Sphinx's `version` config)
try:
    from importlib.metadata import version as pkg_version
    release = pkg_version("Vortex")   # <-- change to your real distribution name if needed (e.g. "vortex")
except Exception:
    release = "1.0.0"

# Short X.Y version for the sidebar/footer
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",
]


autodoc_mock_imports = [
    "torch",
    "triton",
    "numpy",
    "vortex_torch_C",
    "flash-attn",
    "flashinfer",
    "sglang",
]


autosummary_generate = True
autosummary_generate_overwrite = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# Drop the long ``vortex_torch.indexer.elementwise_binary.`` module-path
# prefix from signature headers — each class lives on its own module page,
# so the bare ``class Maximum(...)`` name is unambiguous and fits the card.
add_module_names = False
# Render type-hint cross-references with their short name but resolve them to
# the exact (fully-qualified) target — disambiguates same-named classes that
# legitimately exist in two packages (e.g. cache.Context vs indexer.Context).
python_use_unqualified_type_names = True
autodoc_default_options = {
    #"members": True,
    "undoc-members": False,
    "show-inheritance": True,
    # ``profile`` is the trace-time graph-registration hook — an internal
    # implementation detail, not part of the user-facing op surface. Hide it
    # everywhere so the rendered op pages show only the Math / __init__ /
    # __call__ / Note contract from the class docstring.
    "exclude-members": "profile",
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
# NOTE: "linkify" is intentionally omitted — it needs the extra `linkify-it-py`
# dependency (absent in the docs CI) and we use explicit Markdown links anyway.
myst_enable_extensions = ["deflist", "substitution", "tasklist"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "landing", "Thumbs.db", ".DS_Store"]
language = "en"

# -- HTML --------------------------------------------------------------------
html_theme = "furo"
html_title = f"{project} Documentation"
html_static_path = ["_static"]
html_favicon = "_static/vortex_mark.png"
# Vortex mark in the sidebar (same image for light & dark furo modes).
html_theme_options = {
    "light_logo": "vortex_mark.png",
    "dark_logo": "vortex_mark.png",
}

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),  # <- None instead of {}
}

todo_include_todos = True

def setup(app):
    app.add_css_file("custom.css")