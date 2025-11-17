# qtlite/examples/single_factor_backtest.py
# -*- coding: utf-8 -*-
"""
单因子多空回测 Demo

- 构造一份 toy price 数据
- 人为造一个和未来收益略相关的因子
- 用 FactorLongShortStrategy 做多空
- 用 SimpleBacktestEngine 回测
- 打印指标，并保存净值曲线 PNG
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qtlite.strategy import FactorLongShortConfig, FactorLongShortStrategy
from qtlite.backtest.engine import SimpleBacktestEngine, EngineConfig


def make_toy_prices(n_days: int = 250, n_assets: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(123)

    data = {}
    for i in range(n_assets):
        name = f"Stock{i:02d}"
        # 简单随机游走 + 不同的轻微趋势
        drift = 0.0005 * (i - n_assets / 2)
        ret = drift + rng.normal(0, 0.02, size=n_days)
        price = 10 * (1 + pd.Series(ret, index=dates)).cumprod()
        data[name] = price

    prices = pd.DataFrame(data, index=dates)
    prices.index.name = "date"
    return prices


def make_predictive_factor(returns: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    构造一个跟未来收益有轻微正相关的因子：
    - 因子 ≈ 未来 horizon 日收益 + 一点噪声
    """
    fwd = returns.shift(-1).rolling(horizon).sum()
    rng = np.random.default_rng(456)
    noise = rng.normal(0, 0.01, size=fwd.shape)
    factor = fwd + noise
    factor.columns = returns.columns
    factor.index.name = "date"
    return factor


def main():
    # 1) 构造数据
    prices = make_toy_prices(n_days=250, n_assets=10)
    returns = prices.pct_change().fillna(0.0)

    factor = make_predictive_factor(returns, horizon=3)

    # 2) 配置策略：5 分组，多空对冲
    cfg = FactorLongShortConfig(
        factor_name="alpha",
        n_quantiles=5,
        gross_leverage=1.0,
        min_names_per_side=2,
    )
    strategy = FactorLongShortStrategy(cfg)

    # 3) 回测引擎
    engine = SimpleBacktestEngine(
        EngineConfig(
            transaction_cost=0.0005,  # 单边手续费
        )
    )

    # 4) 运行回测
    result = engine.run(
        prices=prices,
        strategy=strategy,
        risk_model=None,
        constraints=None,
        factors={"alpha": factor},
        context=None,   # 我们现在还没用 context，传 None 即可
    )

    # 5) 打印指标
    print("=== Single Factor Long-Short Backtest ===")
    for k, v in result.metrics.items():
        print(f"{k:15s}: {v:.4f}" if isinstance(v, (int, float)) else f"{k:15s}: {v}")

    # 6) 画净值曲线（存成 PNG）
    nav = result.nav
    plt.figure(figsize=(8, 4))
    plt.plot(nav.index, nav.values)
    plt.title("Factor Long-Short NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("single_factor_nav.png", dpi=150)
    print("Saved NAV curve to single_factor_nav.png")


if __name__ == "__main__":
    main()
