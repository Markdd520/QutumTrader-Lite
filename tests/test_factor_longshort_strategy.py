# tests/test_factor_longshort_strategy.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from qtlite.strategy import FactorLongShortConfig, FactorLongShortStrategy
from qtlite.backtest.engine import SimpleBacktestEngine, EngineConfig
from qtlite.core.enums import PositionSide, Frequency
from qtlite.core.interfaces import StrategyContext


def test_factor_longshort_weights_shape(toy_factor):
    cfg = FactorLongShortConfig(
        factor_name="alpha",
        n_quantiles=5,
        gross_leverage=1.0,
        min_names_per_side=1,
    )
    strat = FactorLongShortStrategy(cfg)

    factors = {"alpha": toy_factor}
    ctx = None  # 我们现在的策略实现不会用到 context

    w = strat.generate_signals(factors, ctx)


    # 形状应该和因子一样
    assert w.shape == toy_factor.shape
    assert not w.isna().any().any()

    net = w.sum(axis=1)
    assert np.nanmedian(np.abs(net)) < 1e-6


def test_factor_longshort_backtest_integration(toy_prices, toy_factor):
    cfg = FactorLongShortConfig(
        factor_name="alpha",
        n_quantiles=5,
        gross_leverage=1.0,
        min_names_per_side=2,
    )
    strat = FactorLongShortStrategy(cfg)

    ctx = None  # 简化：目前 engine 也没用 context 里的字段

    engine = SimpleBacktestEngine(
        EngineConfig(transaction_cost=0.0005)
    )

    result = engine.run(
        prices=toy_prices,
        strategy=strat,
        risk_model=None,
        constraints=None,
        factors={"alpha": toy_factor},
        context=ctx,
    )
