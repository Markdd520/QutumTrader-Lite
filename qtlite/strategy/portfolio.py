# qtlite/strategy/portfolio.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    """计算单日换手率"""
    prev, new = prev_w.align(new_w, join="outer", fill_value=0)
    return 0.5 * (prev - new).abs().sum()


def apply_transaction_cost(
    daily_ret: float,
    turnover: float,
    cost: float = 0.0005,
) -> float:
    """
    将交易成本作用于组合收益：
    cost 为单边费率（如万分之 5 = 0.0005）
    """
    return daily_ret - turnover * cost


def rebalance(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost: float,
) -> pd.Series:
    """
    一天的组合回报（给 engine 调用）：
        r_t = sum( w_{t-1} * ret_t )
    """
    wsp = target_weights.shift(1).reindex(returns.index).fillna(0)
    raw = (wsp * returns).sum(axis=1)

    # turnover & cost
    turn = wsp.diff().abs().sum(axis=1) * 0.5
    net = raw - turn * cost
    return net
