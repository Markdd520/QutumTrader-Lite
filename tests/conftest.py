# tests/conftest.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_assets():
    return ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def toy_dates():
    return pd.date_range("2020-01-01", periods=40, freq="B")  # 40 个交易日


@pytest.fixture
def toy_prices(toy_dates, toy_assets):
    """
    构造一个简单的价格矩阵：
    - AAA/BBB: 稍微有上升趋势
    - CCC/DDD: 横盘
    - EEE: 稍微下跌
    """
    n = len(toy_dates)
    base = np.linspace(10, 12, n)  # 趋势
    rng = np.random.default_rng(123)

    data = {}
    data["AAA"] = base + rng.normal(0, 0.1, n)
    data["BBB"] = base * 1.05 + rng.normal(0, 0.1, n)
    data["CCC"] = np.full(n, 10.0) + rng.normal(0, 0.05, n)
    data["DDD"] = np.full(n, 9.8) + rng.normal(0, 0.05, n)
    data["EEE"] = base[::-1] + rng.normal(0, 0.1, n)  # 反向

    df = pd.DataFrame(data, index=toy_dates)
    df.index.name = "date"
    return df


@pytest.fixture
def toy_returns(toy_prices):
    return toy_prices.pct_change().fillna(0.0)


@pytest.fixture
def toy_factor(toy_returns):
    """
    构造一个跟未来收益有一点正相关的因子：
    - 用未来 3 日收益平滑，再加一点噪声
    """
    fwd3 = toy_returns.shift(-1).rolling(3).sum()
    rng = np.random.default_rng(456)
    noise = rng.normal(0, 0.01, size=fwd3.shape)
    factor = fwd3 + noise
    factor.columns = toy_returns.columns
    return factor


@pytest.fixture
def toy_industry(toy_assets):
    """
    行业映射：
    - AAA, BBB -> Sec1
    - CCC, DDD -> Sec2
    - EEE      -> Sec3
    """
    sectors = ["Sec1", "Sec1", "Sec2", "Sec2", "Sec3"]
    return pd.Series(sectors, index=toy_assets, name="industry")
