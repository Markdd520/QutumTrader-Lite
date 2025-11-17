# qtlite/backtest/portfolio_bt.py
# -*- coding: utf-8 -*-
"""
Portfolio-level backtest helpers:
- 已有权重 + 收益矩阵 -> 组合收益 & NAV & 指标
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd

from qtlite.core.metrics import (
    ann_return,
    ann_vol,
    sharpe_ratio,
    max_drawdown,
    information_ratio,
)
from qtlite.core.utils import align_panels


@dataclass
class PortfolioBTResult:
    nav: pd.Series
    returns: pd.Series
    turnover: pd.Series
    metrics: Dict[str, float]


def backtest_from_weights(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    benchmark_prices: Optional[pd.DataFrame] = None,
    transaction_cost: float = 0.0005,
    slippage: float = 0.0,
    periods_per_year: int = 252,
) -> PortfolioBTResult:
    """
    用“已给定权重”来做回测的简易接口。

    Parameters
    ----------
    prices : date × asset 收盘价
    weights : date × asset 权重（总和可以不为 1）
    benchmark_prices : 可选，date × 1 的基准指数价格
    transaction_cost : 单边交易成本
    slippage : 滑点
    periods_per_year : 年化基数

    Returns
    -------
    PortfolioBTResult
    """
    prices, weights = align_panels([prices, weights])
    prices = prices[0]
    weights = weights[1]

    rets = prices.pct_change().fillna(0.0)

    # 对齐权重 & 收益
    weights, rets = align_panels([weights, rets])
    weights = weights[0]
    rets = rets[1]

    # 允许权重和不为 1，这里不做归一化；如需可在上游处理。
    gross_ret = (weights * rets).sum(axis=1)

    prev_w = weights.shift(1).fillna(0.0)
    turnover = 0.5 * (weights - prev_w).abs().sum(axis=1)

    cost_rate = transaction_cost + slippage
    net_ret = gross_ret - cost_rate * turnover

    nav = (1.0 + net_ret).cumprod()

    # 绩效指标
    metrics: Dict[str, float] = {}
    metrics["ann_return"] = ann_return(net_ret, periods_per_year)
    metrics["ann_vol"] = ann_vol(net_ret, periods_per_year)
    metrics["sharpe"] = sharpe_ratio(net_ret, periods_per_year=periods_per_year)
    metrics["max_drawdown"] = max_drawdown(nav)

    if benchmark_prices is not None:
        bench_ret = benchmark_prices.iloc[:, 0].pct_change().reindex(net_ret.index)
        metrics["information_ratio"] = information_ratio(
            net_ret,
            bench_ret,
            periods_per_year=periods_per_year,
        )

    return PortfolioBTResult(
        nav=nav,
        returns=net_ret,
        turnover=turnover,
        metrics=metrics,
    )
