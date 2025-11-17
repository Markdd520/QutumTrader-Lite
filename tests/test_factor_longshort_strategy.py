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
    ctx = StrategyContext(
        freq=Frequency.DAILY,
        side=PositionSide.LONG_SHORT,
        benchmark=None,
    )

    w = strat.generate_signals(factors, ctx)

    # 形状应该和因子一样
    assert w.shape == toy_factor.shape
    # 没有 NaN
    assert not w.isna().any().any()

    # 大部分日期净暴露应该接近 0（多空对冲）
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

    ctx = StrategyContext(
        freq=Frequency.DAILY,
        side=PositionSide.LONG_SHORT,
        benchmark=None,
    )

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

    # 基本 sanity check
    assert isinstance(result.nav, pd.Series)
    assert len(result.nav) == len(toy_prices)

    # 指标里应该有 Sharpe / ann_return / max_drawdown 等
    metrics = result.metrics
    for key in ["ann_return", "ann_vol", "sharpe", "max_drawdown"]:
        assert key in metrics
        assert np.isfinite(metrics[key]) or np.isnan(metrics[key])
