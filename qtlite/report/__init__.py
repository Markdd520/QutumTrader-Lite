# qtlite/report/__init__.py

from .plots import (
    make_nav_figure,
    make_ic_figure,
    make_group_nav_figure,
)
from .html_report import (
    render_single_factor_html,
    render_backtest_html,
)
from .markdown_report import (
    single_factor_markdown,
    backtest_markdown,
)

__all__ = [
    "make_nav_figure",
    "make_ic_figure",
    "make_group_nav_figure",
    "render_single_factor_html",
    "render_backtest_html",
    "single_factor_markdown",
    "backtest_markdown",
]
