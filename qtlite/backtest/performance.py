# qtlite/backtest/performance.py
# -*- coding: utf-8 -*-
"""
Performance utilities:
- 回测指标汇总
- 因子 IC / RankIC 时间序列
- 分组收益统计
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from qtlite.core.metrics import (
    ann_return,
    ann_vol,
    sharpe_ratio,
    max_drawdown,
    information_ratio,
    ic as _ic_single,
    rank_ic as _ric_single,
)
from qtlite.core.utils import align_panels


@dataclass
class PerfSummary:
    """
    通用绩效摘要结构，可在报告模块直接使用。
    """
    ann_return: float
    ann_vol: float
    sharpe: float
    max_dd: float
    information_ratio: float | None = None


def summarize_performance(
    ret: pd.Series,
    benchmark_ret: pd.Series | None = None,
    periods_per_year: int = 252,
) -> PerfSummary:
    """
    对组合收益序列做整体指标汇总。
    """
    ret = ret.dropna()
    bench = benchmark_ret.dropna() if benchmark_ret is not None else None

    ar = ann_return(ret, periods_per_year)
    av = ann_vol(ret, periods_per_year)
    sr = sharpe_ratio(ret, periods_per_year=periods_per_year)
    mdd = max_drawdown((1 + ret).cumprod())

    ir = None
    if bench is not None and not bench.empty:
        bench = bench.reindex(ret.index)
        ir = information_ratio(ret, bench, periods_per_year)

    return PerfSummary(
        ann_return=ar,
        ann_vol=av,
        sharpe=sr,
        max_dd=mdd,
        information_ratio=ir,
    )


def factor_ic_series(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """
    计算因子的 IC / RankIC 时间序列。
    - factor, fwd_ret: date × asset 宽表（已对齐）
    """
    factor, fwd_ret = align_panels([factor, fwd_ret])
    factor = factor[0]
    fwd_ret = fwd_ret[1]

    ic_list: List[float] = []
    ric_list: List[float] = []
    idx: List[pd.Timestamp] = []

    for dt in factor.index:
        f = factor.loc[dt]
        r = fwd_ret.loc[dt]
        ic_val = _ic_single(f, r)
        ric_val = _ric_single(f, r)
        ic_list.append(ic_val)
        ric_list.append(ric_val)
        idx.append(dt)

    ic_s = pd.Series(ic_list, index=idx, name="IC")
    ric_s = pd.Series(ric_list, index=idx, name="RankIC")
    return ic_s, ric_s


def quantile_group_returns(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quantiles: int = 5,
    long_short: bool = True,
    weights: str = "equal",
) -> Dict[str, pd.Series]:
    """
    按因子分组，计算每一组的等权/市值权重收益，以及多空组合。
    - factor / fwd_ret: date × asset
    - weights: "equal" 目前只实现等权，后面可以扩展为 "vw" 等。
    """
    factor, fwd_ret = align_panels([factor, fwd_ret])
    factor = factor[0]
    fwd_ret = fwd_ret[1]

    q_rets: Dict[int, pd.Series] = {q: [] for q in range(1, n_quantiles + 1)}
    dates: List[pd.Timestamp] = []

    for dt in factor.index:
        f = factor.loc[dt]
        r = fwd_ret.loc[dt]
        df = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
        if df.empty:
            # 用 NaN 占位，保持时间索引一致
            for q in q_rets:
                q_rets[q].append(np.nan)
            dates.append(dt)
            continue

        # 分组
        df["group"] = pd.qcut(
            df["f"].rank(method="first"),
            q=n_quantiles,
            labels=False,
        ) + 1  # 1..n_quantiles

        grouped = df.groupby("group")

        if weights == "equal":
            g_ret = grouped["r"].mean()
        else:
            # 预留扩展：市值加权等
            g_ret = grouped["r"].mean()

        for q in range(1, n_quantiles + 1):
            q_rets[q].append(float(g_ret.get(q, np.nan)))

        dates.append(dt)

    # 转换为时间序列
    out: Dict[str, pd.Series] = {}
    for q in range(1, n_quantiles + 1):
        out[f"Q{q}"] = pd.Series(q_rets[q], index=dates)

    if long_short:
        # 多空 = 最高组 - 最低组
        out["long_short"] = out[f"Q{n_quantiles}"] - out["Q1"]

    return out
