# qtlite/backtest/engine.py
# -*- coding: utf-8 -*-
"""
Simple backtest engine (Lite)
- 支持任意 Strategy / RiskModel / RiskConstraint
- 执行：权重 -> 组合收益 -> NAV -> 指标

目标：
对标私募的“研究框架回测内核”，但先给出一个
结构清晰、逻辑完整的简化版实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from qtlite.core.interfaces import (
    Strategy,
    StrategyContext,
    RiskModel,
    RiskConstraint,
    BacktestResult,
)
from qtlite.core.metrics import (
    ann_return,
    ann_vol,
    sharpe_ratio,
    max_drawdown,
)
from qtlite.core.enums import SignalType, PositionSide
from qtlite.core.utils import align_panels


DataPanel = pd.DataFrame


@dataclass
class EngineConfig:
    """
    回测引擎配置（可以之后映射到 JSON/YAML）

    Attributes
    ----------
    transaction_cost : float
        单边交易成本（例如 0.0005 = 5bp）。
    slippage : float
        简单滑点假设（按成交金额比例）。
    rebalance_lag : int
        调仓滞后：1 表示今天生成的权重在下一个交易日生效。
    periods_per_year : int
        年化基数，默认 252（日频）。
    """

    transaction_cost: float = 0.0005
    slippage: float = 0.0
    rebalance_lag: int = 1
    periods_per_year: int = 252


class SimpleBacktestEngine:
    """
    极简的日频回测引擎：
    - 使用收盘价计算简单收益
    - 权重乘以未来一天收益
    - 支持：Strategy + 可选 RiskModel + 多个 RiskConstraint
    """

    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()

    # ========= 对外唯一入口 =========

    def run(
        self,
        prices: DataPanel,   # 收盘价 date × asset
        strategy: Strategy,
        risk_model: Optional[RiskModel],
        constraints: Optional[Mapping[str, RiskConstraint]],
        factors: Mapping[str, DataPanel],
        context: StrategyContext,
        benchmark_prices: Optional[DataPanel] = None,
    ) -> BacktestResult:
        """
        Run backtest.

        Parameters
        ----------
        prices : DataPanel
            Close prices (date × asset).
        strategy : Strategy
            实现了 generate_signals 的策略对象。
        risk_model : RiskModel, optional
            风险模型，用于给 risk constraints 提供 risk_state。
        constraints : Mapping[str, RiskConstraint], optional
            约束字典：name -> RiskConstraint 实例，按顺序应用。
        factors : Mapping[str, DataPanel]
            策略需要用到的因子字典。
        context : StrategyContext
            策略上下文（频率、方向、benchmark 等）。
        benchmark_prices : DataPanel, optional
            基准价格（如指数），用于计算 IR 等。

        Returns
        -------
        BacktestResult
        """

        # 1) 对齐所有 panel
        aligned = align_panels([prices] + list(factors.values()))
        prices = aligned[0]
        factor_names = list(factors.keys())
        factors_aligned = dict(zip(factor_names, aligned[1:]))

        # 2) 计算收益矩阵（简单收益）
        rets = prices.pct_change().fillna(0.0)

        # 3) 生成策略信号 / 目标权重
        raw_signals = strategy.generate_signals(
            factors=factors_aligned,
            context=context,
        )

        # 对齐信号 & 收益
        raw_signals, rets = align_panels([raw_signals, rets])
        raw_signals = raw_signals[0]
        rets = rets[1]

        # 4) 将 signal 转换为权重（若本身就是 WEIGHT 就直接用）
        weights = self._signals_to_weights(raw_signals, strategy, context)

        # 5) 调仓滞后：权重向后 shift
        lag = max(self.cfg.rebalance_lag, 0)
        if lag > 0:
            weights = weights.shift(lag)

        # 6) 估计风险（如果给了 risk_model）
        if risk_model is not None:
            # 简化：用全样本收益估计一个全局风险状态
            risk_state = risk_model.estimate_risk(rets)
        else:
            risk_state = None

        # 7) 应用风控约束（逐日 / 全局）
        if constraints:
            for _, cons in constraints.items():
                weights = cons.apply(weights, risk_state)

        # 8) 计算组合收益（包含交易成本）
        port_ret, turnover = self._portfolio_returns(
            weights=weights,
            rets=rets,
        )

        # 9) 生成 NAV
        nav = (1.0 + port_ret).cumprod()

        # 10) 计算指标
        metrics = self._compute_metrics(
            port_ret,
            nav,
            benchmark_prices,
        )

        extra: Dict[str, object] = {
            "weights": weights,
            "asset_returns": rets,
            "turnover_series": turnover,
        }

        return BacktestResult(
            nav=nav,
            returns=port_ret,
            turnover=turnover,
            metrics=metrics,
            extra=extra,
        )

    # ========= 内部工具函数 =========

    def _signals_to_weights(
        self,
        signals: DataPanel,
        strategy: Strategy,
        context: StrategyContext,
    ) -> DataPanel:
        """
        把策略输出的 signal 统一转成权重。
        - 如果 signal_type = WEIGHT：直接返回
        - SCORE / RANK：做截面标准化 & 方向控制
        """
        if strategy.signal_type == SignalType.WEIGHT:
            w = signals.copy()
        else:
            # 截面去均值 / 除以绝对值和，构造多空权重
            def _to_w(s: pd.Series) -> pd.Series:
                s = s.replace(np.nan, 0.0)
                if strategy.signal_type == SignalType.RANK:
                    s = s.rank()
                s = s - s.mean()
                denom = np.abs(s).sum()
                if denom == 0:
                    return s * 0.0
                return s / denom

            w = signals.apply(_to_w, axis=1)

        # 如果是 LONG_ONLY，截断负权重并再归一化
        if context.side == PositionSide.LONG_ONLY:
            def _long_only(s: pd.Series) -> pd.Series:
                s = s.clip(lower=0.0)
                total = s.sum()
                if total == 0:
                    return s
                return s / total

            w = w.apply(_long_only, axis=1)

        return w

    def _portfolio_returns(
        self,
        weights: DataPanel,
        rets: DataPanel,
    ) -> tuple[pd.Series, pd.Series]:
        """
        组合收益 & 换手率
        - 当前权重 * 当期个股收益 -> 组合收益
        - turnover ≈ 0.5 * sum(|w_t - w_{t-1}|)

        交易成本 & 滑点按换手率线性扣减。
        """
        weights, rets = align_panels([weights, rets])
        weights = weights[0]
        rets = rets[1]

        # 组合收益（不含成本）
        gross_ret = (weights * rets).sum(axis=1)

        # 换手率
        prev_w = weights.shift(1).fillna(0.0)
        turnover = 0.5 * (weights - prev_w).abs().sum(axis=1)

        # 成本 = (交易成本 + 滑点) * 换手率
        cost_rate = self.cfg.transaction_cost + self.cfg.slippage
        net_ret = gross_ret - cost_rate * turnover

        return net_ret, turnover

    def _compute_metrics(
        self,
        port_ret: pd.Series,
        nav: pd.Series,
        benchmark_prices: Optional[DataPanel],
    ) -> Dict[str, float]:
        """
        计算基本绩效指标：年化收益、波动率、Sharpe、最大回撤、IR（如果有 benchmark）。
        """
        pp_year = self.cfg.periods_per_year

        m: Dict[str, float] = {}
        m["ann_return"] = ann_return(port_ret, pp_year)
        m["ann_vol"] = ann_vol(port_ret, pp_year)
        m["sharpe"] = sharpe_ratio(port_ret, periods_per_year=pp_year)
        m["max_drawdown"] = max_drawdown(nav)

        # 可选：基准 IR
        if benchmark_prices is not None:
            bench_ret = benchmark_prices.iloc[:, 0].pct_change().reindex(
                port_ret.index
            )
            from qtlite.core.metrics import information_ratio

            m["information_ratio"] = information_ratio(
                port_ret,
                bench_ret,
                periods_per_year=pp_year,
            )

        return m
