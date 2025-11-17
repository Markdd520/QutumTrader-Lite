# qtlite/risk/exposure.py
# -*- coding: utf-8 -*-
"""
Exposure analysis:
- Sector / industry exposure
- Style factor exposure (simple regression-style)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def sector_exposure(
    weights: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """
    计算组合的行业暴露（逐日）。
    - weights: date × asset 权重
    - sector_map: index=asset, value=sector_code

    返回：date × sector 的行业权重（等于权重和）
    """
    # 对齐资产
    common_cols = weights.columns.intersection(sector_map.index)
    w = weights[common_cols]
    sec = sector_map.loc[common_cols]

    # 每天：按行业分组求和
    def _one_day(s: pd.Series) -> pd.Series:
        df = pd.DataFrame({"w": s, "sec": sec})
        g = df.groupby("sec")["w"].sum()
        return g

    expo = w.apply(_one_day, axis=1)
    expo.index = weights.index
    return expo.fillna(0.0)


def style_exposure(
    weights: pd.DataFrame,
    style_loadings: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算组合对若干 style 因子（如 SIZE, VALUE, MOMENTUM）的暴露。

    参数
    ----
    weights : date × asset 权重
    style_loadings : asset × style 的因子载荷矩阵
        比如：
        index = asset
        columns = ["SIZE", "VALUE", "MOMENTUM", ...]

    返回
    ----
    date × style 的暴露（约等于加权平均载荷）
    """
    # 对齐
    common_assets = weights.columns.intersection(style_loadings.index)
    w = weights[common_assets]
    B = style_loadings.loc[common_assets]  # asset × style

    # 暴露 ~ wᵀ * B
    # 每天：expo_t = w_t(1×N) * B(N×K) => 1×K
    expos_list = []
    for dt, w_t in w.iterrows():
        wt = w_t.values.reshape(1, -1)  # (1,N)
        expo_t = wt @ B.values  # (1,K)
        expos_list.append(expo_t.flatten())

    expo_arr = np.vstack(expos_list) if expos_list else np.zeros((0, B.shape[1]))
    expo_df = pd.DataFrame(
        expo_arr,
        index=w.index,
        columns=B.columns,
    )
    return expo_df
