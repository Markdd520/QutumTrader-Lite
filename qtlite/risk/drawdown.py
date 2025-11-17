# qtlite/risk/drawdown.py
# -*- coding: utf-8 -*-
"""
Drawdown & tail risk utilities:
- rolling max drawdown
- CVaR (expected shortfall)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from qtlite.core.metrics import drawdown_curve


def rolling_max_drawdown(
    nav: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    滚动窗口最大回撤。
    """
    nav = nav.dropna()
    if nav.empty:
        return pd.Series(dtype=float)

    dd_list = []
    idx = []

    for i in range(window, len(nav) + 1):
        sub = nav.iloc[i - window : i]
        _, dd = drawdown_curve(sub)
        dd_list.append(dd.min())
        idx.append(sub.index[-1])

    return pd.Series(dd_list, index=idx, name="rolling_max_drawdown")


def cvar(
    ret: pd.Series,
    alpha: float = 0.95,
) -> float:
    """
    CVaR (Expected Shortfall) at level alpha:
    - 假设 ret 为日收益
    - CVaR = 条件期望损失（在最差 (1-alpha)% 情形下的平均收益）

    注意：返回的是一个负数（平均损失）。
    """
    ret = ret.dropna()
    if ret.empty:
        return np.nan

    q = ret.quantile(1 - alpha)
    tail = ret[ret <= q]
    if tail.empty:
        return np.nan
    return float(tail.mean())


def rolling_cvar(
    ret: pd.Series,
    window: int = 252,
    alpha: float = 0.95,
) -> pd.Series:
    """
    滚动 CVaR。
    """
    ret = ret.dropna()
    if ret.empty:
        return pd.Series(dtype=float)

    vals = []
    idx = []

    for i in range(window, len(ret) + 1):
        sub = ret.iloc[i - window : i]
        vals.append(cvar(sub, alpha=alpha))
        idx.append(sub.index[-1])

    return pd.Series(vals, index=idx, name="rolling_cvar")
