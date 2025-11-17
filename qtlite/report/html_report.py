# qtlite/report/html_report.py
# -*- coding: utf-8 -*-
"""
HTML report generator (Jinja2 + Plotly)
- 单因子报告
- 策略回测报告
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from jinja2 import Template

from qtlite.backtest.single_factor_bt import SingleFactorResult
from qtlite.core.interfaces import BacktestResult
from .plots import (
    make_nav_figure,
    make_ic_figure,
    make_group_nav_figure,
)


_SINGLE_FACTOR_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ factor_name }} - Single Factor Report</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Microsoft YaHei", "PingFang SC",
                   Roboto, "Helvetica Neue", Arial, sans-serif;
      max-width: 1100px;
      margin: 2rem auto;
      padding: 0 1rem 4rem 1rem;
      background-color: #fafafa;
      color: #222;
    }
    h1, h2, h3 {
      margin-top: 1.8rem;
      margin-bottom: 0.6rem;
    }
    .card {
      background: #fff;
      border-radius: 10px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1.2rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 0.9rem;
    }
    th, td {
      border: 1px solid #ddd;
      padding: 0.35rem 0.5rem;
      text-align: right;
    }
    th {
      background: #f2f2f2;
    }
    th:first-child, td:first-child {
      text-align: left;
    }
    .section-title {
      margin-bottom: 0.5rem;
    }
  </style>
</head>
<body>
  <h1>Single Factor Report: {{ factor_name }}</h1>

  <div class="card">
    <h2 class="section-title">Summary Statistics (Long-Short)</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
      {% for k, v in metrics.items() %}
        <tr>
          <td>{{ k }}</td>
          <td>{{ "{:.4f}".format(v) if v == v else "NaN" }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2 class="section-title">Group NAV (Q1..Qn, long_short)</h2>
    {{ group_nav_html | safe }}
  </div>

  <div class="card">
    <h2 class="section-title">IC / RankIC Time Series</h2>
    {{ ic_html | safe }}
  </div>

</body>
</html>
"""


_BACKTEST_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ strategy_name }} - Backtest Report</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Microsoft YaHei", "PingFang SC",
                   Roboto, "Helvetica Neue", Arial, sans-serif;
      max-width: 1100px;
      margin: 2rem auto;
      padding: 0 1rem 4rem 1rem;
      background-color: #fafafa;
      color: #222;
    }
    h1, h2, h3 {
      margin-top: 1.8rem;
      margin-bottom: 0.6rem;
    }
    .card {
      background: #fff;
      border-radius: 10px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1.2rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 0.9rem;
    }
    th, td {
      border: 1px solid #ddd;
      padding: 0.35rem 0.5rem;
      text-align: right;
    }
    th {
      background: #f2f2f2;
    }
    th:first-child, td:first-child {
      text-align: left;
    }
    .section-title {
      margin-bottom: 0.5rem;
    }
  </style>
</head>
<body>
  <h1>Backtest Report: {{ strategy_name }}</h1>

  <div class="card">
    <h2 class="section-title">Summary Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
      {% for k, v in metrics.items() %}
        <tr>
          <td>{{ k }}</td>
          <td>{{ "{:.4f}".format(v) if v == v else "NaN" }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2 class="section-title">NAV Curve</h2>
    {{ nav_html | safe }}
  </div>

  {% if ic_html %}
  <div class="card">
    <h2 class="section-title">IC / RankIC (if provided)</h2>
    {{ ic_html | safe }}
  </div>
  {% endif %}

</body>
</html>
"""


def render_single_factor_html(
    result: SingleFactorResult,
    out_path: str | Path,
) -> None:
    """
    生成单因子 HTML 报告。
    """
    out_path = Path(out_path)

    # Group NAV 图
    fig_nav = make_group_nav_figure(result.group_nav)
    group_nav_html = fig_nav.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )

    # IC 图
    fig_ic = make_ic_figure(result.ic, result.rank_ic)
    ic_html = fig_ic.to_html(
        full_html=False,
        include_plotlyjs=False,
    )

    tpl = Template(_SINGLE_FACTOR_TEMPLATE)
    html = tpl.render(
        factor_name=result.factor_name,
        metrics=result.metrics_long_short,
        group_nav_html=group_nav_html,
        ic_html=ic_html,
    )

    out_path.write_text(html, encoding="utf-8")


def render_backtest_html(
    result: BacktestResult,
    strategy_name: str,
    out_path: str | Path,
    benchmark_nav: Optional[pd.Series] = None,
    ic_series: Optional[pd.Series] = None,
    rank_ic_series: Optional[pd.Series] = None,
) -> None:
    """
    生成策略回测 HTML 报告。

    Parameters
    ----------
    result : BacktestResult
        回测引擎产出的结果。
    strategy_name : str
        策略名称。
    out_path : str | Path
        输出 HTML 路径。
    benchmark_nav : pd.Series, optional
        基准净值，用于 NAV 对比。
    ic_series / rank_ic_series : 可选
        若策略是因子驱动，可以传入 IC 序列一起展示。
    """
    out_path = Path(out_path)

    # NAV 图
    fig_nav = make_nav_figure(
        nav=result.nav,
        benchmark_nav=benchmark_nav,
        title=f"{strategy_name} NAV",
    )
    nav_html = fig_nav.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )

    # IC 图（可选）
    if ic_series is not None:
        fig_ic = make_ic_figure(
            ic_series,
            rank_ic_series,
            title="IC / RankIC",
        )
        ic_html = fig_ic.to_html(
            full_html=False,
            include_plotlyjs=False,
        )
    else:
        ic_html = ""

    tpl = Template(_BACKTEST_TEMPLATE)
    html = tpl.render(
        strategy_name=strategy_name,
        metrics=result.metrics,
        nav_html=nav_html,
        ic_html=ic_html,
    )

    out_path.write_text(html, encoding="utf-8")
