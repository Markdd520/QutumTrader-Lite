# qtlite/data_factor/loaders.py
# -*- coding: utf-8 -*-
"""
Data loaders for panels:
- CSV / Parquet / demo data
- Unified panel format: date × asset
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def load_csv_panel(
    path: str | Path,
    date_col: str = "date",
    index_name: str = "date",
) -> pd.DataFrame:
    """
    加载宽表 CSV（适合价格、成交量、收益率等数据）
    要求：列格式如下：

        date, 000001.SZ, 000002.SZ, ...
        2020-01-01, 12.3,  7.1, ...

    Returns
    -------
    DataFrame
        index = datetime (date)
        columns = asset codes
    """
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"CSV must contain '{date_col}' column")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = index_name

    # All remaining columns are assets
    return df.sort_index()


def load_parquet_panel(
    path: str | Path,
    index_name: str = "date",
) -> pd.DataFrame:
    """
    直接读取 parquet 宽表。

    你未来如果用到更大的全 A 数据，可以扩展 schema。
    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = index_name
    return df.sort_index()


# ========= Demo Data =========

def load_demo_data() -> Dict[str, pd.DataFrame]:
    """
    载入内置示例数据（供 GitHub repo 可直接运行）。

    包含：
    - prices.csv    收盘价面板
    - volume.csv    成交量
    - vwap.csv      加权平均成交价
    - industry.csv  股票 → 行业映射
    """
    base = Path(__file__).resolve().parent / "demo_data"

    prices = load_csv_panel(base / "prices.csv")
    volume = load_csv_panel(base / "volume.csv")
    vwap = load_csv_panel(base / "vwap.csv")

    industry = pd.read_csv(
        base / "industry.csv",
        index_col=0,
        squeeze=True,
    )

    return {
        "prices": prices,
        "volume": volume,
        "vwap": vwap,
        "industry": industry,
    }
