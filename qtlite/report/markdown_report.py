# qtlite/report/markdown_report.py
# -*- coding: utf-8 -*-
"""
Markdown report helpers:
- Single factor summary
- Backtest summary
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from qtlite.backtest.single_factor_bt import SingleFactorResult
from qtlite.core.interfaces import BacktestResult


def _format_float(v: float, ndigits: int = 4) -> str:
    if v != v:  # NaN
        return "NaN"
    return f"{v:.{ndigits}f}"


def single_factor_markdown(
    result: SingleFactorResult,
    periods_per_year: int = 252,
) -> str:
    """
    生成单因子回测的 Markdown 文本摘要。
    """
    lines: list[str] = []

    lines.append(f"# Single Factor Report: `{result.factor_name}`")
    lines.append("")
    lines.append("## Long-Short Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    for k, v in result.metrics_long_short.items():
        lines.append(f"| {k} | {_format_float(v)} |")

    lines.append("")
    lines.append("## IC / RankIC")
    lines.append("")
    lines.append(f"- Mean IC: `{_format_float(result.metrics_long_short.get('ic_mean', float('nan')) )}`")
    lines.append(f"- Std IC: `{_format_float(result.metrics_long_short.get('ic_std', float('nan')) )}`")
    lines.append(f"- Mean RankIC: `{_format_float(result.metrics_long_short.get('rank_ic_mean', float('nan')) )}`")
    lines.append(f"- Std RankIC: `{_format_float(result.metrics_long_short.get('rank_ic_std', float('nan')) )}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Group NAV and time series plots are available in the HTML report.")
    lines.append("- This summary focuses on the long-short portfolio built from the factor quantiles.")

    return "\n".join(lines)


def backtest_markdown(
    result: BacktestResult,
    strategy_name: str,
    extra_notes: Optional[str] = None,
) -> str:
    """
    生成策略回测的 Markdown 文本摘要。
    """
    lines: list[str] = []

    lines.append(f"# Backtest Report: `{strategy_name}`")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    for k, v in result.metrics.items():
        lines.append(f"| {k} | {_format_float(v)} |")

    lines.append("")
    if extra_notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(extra_notes)

    return "\n".join(lines)
