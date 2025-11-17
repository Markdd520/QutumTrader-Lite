# qtlite/backtest/single_factor_bt.py
# -*- coding: utf-8 -*-
"""
Single factor backtest (Lite Alphalens-style)

输入：
- factor_panel (date × asset)
- forward_return_panel (date × asset)

输出：
- 分组收益时间序列
- 多空收益 / NAV
- IC / RankIC 时间序列
- 汇总指标
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from qtlite.core.metrics import (
    ann_return,
    ann_vol,
    sharpe_ratio,
    max_drawdown,
)
from qtlite.core.utils import align_panels
from .performance import (
    factor_ic_series,
    quantile_group_returns,
)


@dataclass
class SingleFactorResult:
    """
    单因子回测结果整体打包，便于报告模块直接使用。
    """
    factor_name: str
    group_returns: Dict[str, pd.Series]  # Q1..Qn, long_short
    group_nav: Dict[str, pd.Series]
    ic: pd.Series
    rank_ic: pd.Series
    metrics_long_short: Dict[str, float]


def run_single_factor_backtest(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    factor_name: str = "factor",
    n_quantiles: int = 5,
    periods_per_year: int = 252,
) -> SingleFactorResult:
    """
    执行一个完整的单因子回测：
    - 分组收益 / 多空
    - 多空净值
    - IC / RankIC
    - 多空绩效指标
    """
    factor, fwd_ret = align_panels([factor, fwd_ret])
    factor = factor[0]
    fwd_ret = fwd_ret[1]

    # 1) 分组收益
    grp_ret = quantile_group_returns(
        factor=factor,
        fwd_ret=fwd_ret,
        n_quantiles=n_quantiles,
        long_short=True,
    )

    # 2) 组别 NAV
    grp_nav: Dict[str, pd.Series] = {}
    for name, r in grp_ret.items():
        grp_nav[name] = (1.0 + r.fillna(0.0)).cumprod()

    # 3) IC / RankIC 时间序列
    ic_s, ric_s = factor_ic_series(factor, fwd_ret)

    # 4) 多空指标
    ls_ret = grp_ret["long_short"]
    ls_nav = grp_nav["long_short"]

    m: Dict[str, float] = {}
    m["ann_return"] = ann_return(ls_ret, periods_per_year)
    m["ann_vol"] = ann_vol(ls_ret, periods_per_year)
    m["sharpe"] = sharpe_ratio(ls_ret, periods_per_year=periods_per_year)
    m["max_drawdown"] = max_drawdown(ls_nav)
    m["ic_mean"] = float(ic_s.mean())
    m["ic_std"] = float(ic_s.std(ddof=1))
    m["rank_ic_mean"] = float(ric_s.mean())
    m["rank_ic_std"] = float(ric_s.std(ddof=1))

    return SingleFactorResult(
        factor_name=factor_name,
        group_returns=grp_ret,
        group_nav=grp_nav,
        ic=ic_s,
        rank_ic=ric_s,
        metrics_long_short=m,
    )
