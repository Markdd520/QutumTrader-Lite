# qtlite/strategy/execution_sim.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import pandas as pd


def simple_slippage(order_size: pd.Series, slippage_bps: float = 5) -> pd.Series:
    """
    简单滑点模型：
    order_size: 单位为“权重变化” (delta weight)
    slippage_bps: 滑点基点（5bps = 0.0005）
    """
    return order_size.abs() * (slippage_bps / 10000)


def impact_cost(order_size: pd.Series, liquidity: pd.Series, gamma: float = 1.5) -> pd.Series:
    """
    基于冲击成本的模型：
    cost ≈ |Δw|^gamma / liquidity
    """
    adj_liq = liquidity.replace(0, np.nan).fillna(liquidity.mean())
    return (order_size.abs() ** gamma) / adj_liq
