# qtlite/report/plots.py
# -*- coding: utf-8 -*-
"""
Plot utilities (Plotly):
- NAV curve
- IC / RankIC time series
- Group NAV (Q1..Qn, long_short)
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go


def make_nav_figure(
    nav: pd.Series,
    benchmark_nav: Optional[pd.Series] = None,
    title: str = "Strategy NAV",
) -> go.Figure:
    """
    绘制策略净值曲线，可选叠加基准。
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=nav.index,
            y=nav.values,
            mode="lines",
            name="Strategy",
        )
    )

    if benchmark_nav is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_nav.index,
                y=benchmark_nav.values,
                mode="lines",
                name="Benchmark",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="NAV",
        hovermode="x unified",
    )
    return fig


def make_ic_figure(
    ic: pd.Series,
    rank_ic: Optional[pd.Series] = None,
    title: str = "Factor IC / RankIC",
) -> go.Figure:
    """
    绘制因子 IC / RankIC 时间序列。
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=ic.index,
            y=ic.values,
            name="IC",
        )
    )

    if rank_ic is not None:
        fig.add_trace(
            go.Scatter(
                x=rank_ic.index,
                y=rank_ic.values,
                mode="lines",
                name="RankIC",
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                title="RankIC",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="IC",
        hovermode="x unified",
    )
    return fig


def make_group_nav_figure(
    group_nav: Dict[str, pd.Series],
    title: str = "Quantile Group NAV",
) -> go.Figure:
    """
    绘制分组净值曲线：Q1..Qn + long_short。
    """
    fig = go.Figure()

    for name, nav in group_nav.items():
        fig.add_trace(
            go.Scatter(
                x=nav.index,
                y=nav.values,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="NAV",
        hovermode="x unified",
    )
    return fig
