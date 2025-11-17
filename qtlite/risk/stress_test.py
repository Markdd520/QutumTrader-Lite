# qtlite/risk/stress_test.py
# -*- coding: utf-8 -*-
"""
Stress testing utilities:
- Scenario shock on specific assets / sectors
"""

from __future__ import annotations

from typing import Iterable, Dict

import numpy as np
import pandas as pd


def shock_by_assets(
    weights: pd.Series,
    shock_returns: pd.Series,
) -> float:
    """
    对单个时点的组合做资产级冲击：
    - weights: asset 权重
    - shock_returns: asset 对应的一次性 shock 收益（例如 -0.2）
    返回：组合一次性收益（负数为损失）
    """
    w, r = weights.align(shock_returns, join="left")
    r = r.fillna(0.0)
    return float((w * r).sum())


def shock_by_sector(
    weights: pd.Series,
    sector_map: pd.Series,
    shocked_sectors: Iterable[str],
    shock_ret: float = -0.2,
) -> float:
    """
    对一个时点的组合做行业级冲击：
    - shocked_sectors: 被冲击行业列表
    - shock_ret: 行业内资产统一遭遇的冲击收益（如 -0.3）
    """
    # 找到属于目标行业的资产
    mask = sector_map.isin(list(shocked_sectors))
    assets = sector_map.index[mask]

    shock = pd.Series(
        data=shock_ret,
        index=assets,
    )
    return shock_by_assets(weights, shock)
