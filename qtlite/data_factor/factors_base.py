# qtlite/data_factor/factors_base.py
# -*- coding: utf-8 -*-
"""
Basic factor building utilities:
- rolling mean / std / rank
- delta, delay
- TS/CS helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ========= 时间序列操作 =========

def delay(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """TS 延迟 n 期"""
    return df.shift(n)


def delta(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """df - df.shift(n)"""
    return df - df.shift(n)


def ts_mean(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=min(3, n)).mean()


def ts_std(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=min(3, n)).std()


def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=min(3, n)).apply(
        lambda s: s.rank().iloc[-1],
        raw=False,
    )


def ts_argmax(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=min(3, n)).apply(
        lambda s: np.argmax(s) + 1,
        raw=False,
    )


def ts_argmin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=min(3, n)).apply(
        lambda s: np.argmin(s) + 1,
        raw=False,
    )


def ts_corr(a: pd.DataFrame, b: pd.DataFrame, n: int) -> pd.DataFrame:
    return a.rolling(n, min_periods=min(3, n)).corr(b)


# ========= 截面操作 =========

def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面 rank"""
    return df.rank(axis=1)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score"""
    return (df - df.mean(axis=1).values.reshape(-1, 1)) / df.std(axis=1).values.reshape(-1, 1)


def cs_winsor(df: pd.DataFrame, lower=0.01, upper=0.99) -> pd.DataFrame:
    """截面去极值"""
    def _clip(s):
        return s.clip(s.quantile(lower), s.quantile(upper))
    return df.apply(_clip, axis=1)


# ========= 复合工具 =========

def signed_power(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """保留符号的幂次：sign(x) * |x|^p"""
    return np.sign(df) * (np.abs(df) ** p)
