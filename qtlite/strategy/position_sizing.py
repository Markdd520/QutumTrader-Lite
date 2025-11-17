# qtlite/strategy/position_sizing.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import pandas as pd


def equal_weight(names: list[str], total_exposure: float = 1.0) -> pd.Series:
    """对一组股票等权分配"""
    if len(names) == 0:
        return pd.Series()
    w = total_exposure / len(names)
    return pd.Series({n: w for n in names})


def inv_vol_weight(returns: pd.DataFrame, total_exposure: float = 1.0) -> pd.Series:
    """根据波动率倒数加权（Risk Parity 最简版）"""
    vol = returns.std()
    inv = 1 / (vol + 1e-12)
    w = inv / inv.sum() * total_exposure
    return w


def map_groups_to_weights(groups: pd.Series, mapping: dict[int, float]) -> pd.Series:
    """
    将分组→权重映射，例如：
        groups = [1,2,3,4,5]
        mapping = {5:0.5, 1:-0.5}

    返回一个权重向量。
    """
    w = groups.map(mapping).fillna(0)
    return w
