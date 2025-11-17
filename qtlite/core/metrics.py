# qtlite/core/metrics.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def ic(x: pd.Series, y: pd.Series) -> float:
    """
    Pearson IC between factor values and forward returns.

    x, y should have the same index (cross-section on one date).
    """
    s = pd.concat([x, y], axis=1).dropna()
    if s.shape[0] < 2:
        return np.nan
    return float(s.iloc[:, 0].corr(s.iloc[:, 1]))


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    """
    Spearman Rank IC between factor and forward returns.
    """
    s = pd.concat([x, y], axis=1).dropna()
    if s.shape[0] < 2:
        return np.nan
    return float(s.iloc[:, 0].rank().corr(s.iloc[:, 1].rank()))


def ann_return(ret: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized return from periodic returns.
    """
    ret = ret.dropna()
    if ret.empty:
        return np.nan
    cum = (1.0 + ret).prod()
    years = len(ret) / periods_per_year
    if years <= 0:
        return np.nan
    return float(cum ** (1 / years) - 1)


def ann_vol(ret: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized volatility of periodic returns.
    """
    ret = ret.dropna()
    if ret.empty:
        return np.nan
    return float(ret.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(ret: pd.Series, rf: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio with constant risk-free rate `rf`.
    """
    ret = ret.dropna()
    if ret.empty:
        return np.nan
    excess = ret - rf / periods_per_year
    mu = ann_return(excess, periods_per_year)
    vol = ann_vol(excess, periods_per_year)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(mu / vol)


def drawdown_curve(nav: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Compute running max and drawdown series from NAV.
    """
    nav = nav.dropna()
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    return running_max, dd


def max_drawdown(nav: pd.Series) -> float:
    """
    Maximum drawdown (most negative drawdown).
    """
    _, dd = drawdown_curve(nav)
    if dd.empty:
        return np.nan
    return float(dd.min())


def information_ratio(ret: pd.Series, benchmark_ret: pd.Series,
                      periods_per_year: int = 252) -> float:
    """
    Annualized Information Ratio (IR) vs benchmark.
    """
    aligned = pd.concat([ret, benchmark_ret], axis=1).dropna()
    if aligned.empty:
        return np.nan
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    mu = ann_return(diff, periods_per_year)
    vol = ann_vol(diff, periods_per_year)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(mu / vol)
