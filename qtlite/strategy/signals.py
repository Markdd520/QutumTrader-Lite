# qtlite/strategy/signals.py
# -*- coding: utf-8 -*-
"""
Signal utilities for strategies:
- ranking
- zscore
- winsorization
- weight normalization
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面排名归一化到 [0,1]"""
    return df.rank(axis=1, pct=True)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面标准化 Z-Score"""
    return (df - df.mean(axis=1).values[:, None]) / (df.std(axis=1).values[:, None] + 1e-12)


def winsorize(df: pd.DataFrame, p: float = 0.01) -> pd.DataFrame:
    """按百分位 winsorize"""
    lower = df.quantile(p, axis=1)
    upper = df.quantile(1 - p, axis=1)
    return df.clip(lower=lower, upper=upper, axis=0)


def neutralize(df: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    """
    行业中性化（简单版）：每个行业内部减均值
    df: date×asset
    by: asset→industry
    """
    out = df.copy()
    for dt in df.index:
        s = df.loc[dt]
        for sec, idx in by.groupby(by):
            mask = idx.index
            out.loc[dt, mask] = s[mask] - s[mask].mean()
    return out


def scale_to_gross(weights: pd.DataFrame, gross: float) -> pd.DataFrame:
    """
    将权重缩放到指定 gross。
    gross = sum(|w_i|)
    """
    abs_sum = weights.abs().sum(axis=1)
    scale = gross / (abs_sum + 1e-12)
    return weights.mul(scale, axis=0)


def normalize_long_only(weights: pd.DataFrame) -> pd.DataFrame:
    """Long-only 权重归一化到 1"""
    s = weights.sum(axis=1)
    return weights.div(s.replace(0, 1).values.reshape(-1, 1))
