# qtlite/strategy/factor_longshort.py
# -*- coding: utf-8 -*-
"""
Factor-driven strategies:
- Long/short factor strategy (top vs bottom quantile)
- Long-only top-K factor strategy

中文说明：
- 这些策略直接基于“单一综合因子”做仓位分配，
  适合在研究阶段快速验证一个因子的交易能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from qtlite.core.interfaces import Strategy, StrategyContext
from qtlite.core.enums import SignalType

DataPanel = pd.DataFrame


# ========= Configs =========

@dataclass
class FactorLongShortConfig:
    """
    单因子多空策略配置（截面分组 → top vs bottom）.
    """
    factor_name: str
    n_quantiles: int = 5
    top_quantile: Optional[int] = None
    bottom_quantile: Optional[int] = None
    gross_leverage: float = 1.0
    min_names_per_side: int = 5


@dataclass
class FactorTopKConfig:
    """
    单因子多头策略配置（只做 top K / top quantile）.
    """
    factor_name: str
    top_frac: float = 0.1
    n_quantiles: int = 5
    top_quantile: Optional[int] = None
    gross_exposure: float = 1.0
    min_names: int = 5


# ========= Strategies =========

class FactorLongShortStrategy(Strategy):
    """
    单因子多空策略（top vs bottom）.

    - signal_type = WEIGHT
    - 返回的是“目标权重”矩阵（DataFrame），可以直接给 SimpleBacktestEngine 使用。
    """

    def __init__(self, cfg: FactorLongShortConfig, name: Optional[str] = None):
        super().__init__(name=name or f"FactorLongShort({cfg.factor_name})")
        self.cfg = cfg
        self.signal_type = SignalType.WEIGHT  # 直接输出权重

    def generate_signals(
        self,
        factors: Mapping[str, DataPanel],
        context: StrategyContext | None,
    ) -> DataPanel:
        # 1) 取因子
        if self.cfg.factor_name not in factors:
            raise KeyError(f"Factor '{self.cfg.factor_name}' not found in factors dict")

        f = factors[self.cfg.factor_name]

        # 2) 确保日期有序
        f = f.sort_index()

        # 3) 按日构造权重
        w_list = []
        idx_list = []

        n_q = self.cfg.n_quantiles
        top_q = self.cfg.top_quantile or n_q
        bot_q = self.cfg.bottom_quantile or 1

        gross = float(self.cfg.gross_leverage)
        long_target = gross / 2.0
        short_target = -gross / 2.0

        for dt, s in f.iterrows():
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                # 当日无有效因子，空仓
                w_list.append(pd.Series(dtype=float))
                idx_list.append(dt)
                continue

            # 截面排名
            r = s.rank(method="first")
            try:
                # 分组：1..n_q
                group = pd.qcut(
                    r,
                    q=n_q,
                    labels=False,
                ) + 1
            except ValueError:
                # 样本太少无法分组 → 空仓
                w_list.append(pd.Series(dtype=float))
                idx_list.append(dt)
                continue

            group = pd.Series(group, index=s.index)

            long_names = group[group == top_q].index
            short_names = group[group == bot_q].index

            if (len(long_names) < self.cfg.min_names_per_side) or (
                len(short_names) < self.cfg.min_names_per_side
            ):
                # 当日候选太少，空仓
                w_list.append(pd.Series(dtype=float))
                idx_list.append(dt)
                continue

            # 多头等权：sum = long_target
            w_long = pd.Series(
                data=long_target / len(long_names),
                index=long_names,
            )
            # 空头等权：sum = short_target
            w_short = pd.Series(
                data=short_target / len(short_names),
                index=short_names,
            )

            w_t = pd.concat([w_long, w_short])
            idx_list.append(dt)
            w_list.append(w_t)

        # 4) 组装成 DataFrame（date × asset）
        weights = pd.DataFrame(w_list, index=idx_list).sort_index()
        weights = weights.reindex(columns=f.columns).fillna(0.0)

        return weights


class FactorTopKLongStrategy(Strategy):
    """
    单因子多头策略（top-K / top-quantile）.

    - signal_type = WEIGHT
    - 每日只在“最看好的一批股票”中等权持仓，其余为 0。
    """

    def __init__(self, cfg: FactorTopKConfig, name: Optional[str] = None):
        super().__init__(name=name or f"FactorTopKLong({cfg.factor_name})")
        self.cfg = cfg
        self.signal_type = SignalType.WEIGHT

    def generate_signals(
        self,
        factors: Mapping[str, DataPanel],
        context: StrategyContext | None,
    ) -> DataPanel:
        if self.cfg.factor_name not in factors:
            raise KeyError(f"Factor '{self.cfg.factor_name}' not found in factors dict")

        f = factors[self.cfg.factor_name].sort_index()

        w_list = []
        idx_list = []

        n_q = self.cfg.n_quantiles
        top_q = self.cfg.top_quantile  # 可以为 None
        gross = float(self.cfg.gross_exposure)

        for dt, s in f.iterrows():
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                w_list.append(pd.Series(dtype=float))
                idx_list.append(dt)
                continue

            r = s.rank(method="first")

            if top_q is not None:
                # 分组方式：只取第 top_q 组
                try:
                    group = pd.qcut(
                        r,
                        q=n_q,
                        labels=False,
                    ) + 1
                except ValueError:
                    w_list.append(pd.Series(dtype=float))
                    idx_list.append(dt)
                    continue

                group = pd.Series(group, index=s.index)
                long_names = group[group == top_q].index
            else:
                # 直接按 top_frac 选前 K%
                k = max(int(len(s) * self.cfg.top_frac), self.cfg.min_names)
                # rank 越大，因子越高；这里选 rank 最大的 k 个
                long_names = r.sort_values(ascending=False).index[:k]

            if len(long_names) < self.cfg.min_names:
                w_list.append(pd.Series(dtype=float))
                idx_list.append(dt)
                continue

            # 等权 long-only，总权重为 gross
            w_t = pd.Series(
                data=gross / len(long_names),
                index=long_names,
            )

            w_list.append(w_t)
            idx_list.append(dt)

        weights = pd.DataFrame(w_list, index=idx_list).sort_index()
        weights = weights.reindex(columns=f.columns).fillna(0.0)

        return weights
