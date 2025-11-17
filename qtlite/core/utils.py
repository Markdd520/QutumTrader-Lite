# qtlite/core/utils.py
from __future__ import annotations

from typing import Iterable, Mapping, Dict, Any

import numpy as np
import pandas as pd


def align_panels(panels: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
    """
    Align multiple wide panels on the intersection of
    index (dates) and columns (assets).

    常用来对齐：price, returns, factor_panel 等。
    """
    panels = list(panels)
    if not panels:
        return panels

    # intersect index & columns
    common_index = panels[0].index
    common_cols = panels[0].columns
    for df in panels[1:]:
        common_index = common_index.intersection(df.index)
        common_cols = common_cols.intersection(df.columns)

    aligned = [df.loc[common_index, common_cols] for df in panels]
    return aligned


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Simple cross-sectional winsorization by quantiles (per date).
    """
    def _clip(s: pd.Series) -> pd.Series:
        ql = s.quantile(lower)
        qh = s.quantile(upper)
        return s.clip(ql, qh)

    return df.apply(_clip, axis=1)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score (per date).
    """
    def _zs(s: pd.Series) -> pd.Series:
        mu = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return s * np.nan
        return (s - mu) / std

    return df.apply(_zs, axis=1)


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Safe conversion to numpy array (float64).
    """
    return np.asarray(df, dtype=float)


def ensure_dataframe(obj: Any, index=None, columns=None) -> pd.DataFrame:
    """
    Ensure the input is a DataFrame. Useful in API normalization.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    return pd.DataFrame(obj, index=index, columns=columns)
