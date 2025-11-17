# qtlite/backtest/regime_analysis.py
# -*- coding: utf-8 -*-
"""
Regime analysis:
- 按基准收益/波动率划分市场状态
- 在不同状态下统计策略表现
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import pandas as pd

from qtlite.core.metrics import ann_return, ann_vol, sharpe_ratio
from qtlite.core.utils import align_panels


RegimeLabel = Literal["bull", "bear", "sideways"]


@dataclass
class RegimePerf:
    regime: RegimeLabel
    ann_return: float
    ann_vol: float
    sharpe: float
    sample_size: int


def classify_regime_by_benchmark(
    benchmark_ret: pd.Series,
    up_quantile: float = 0.66,
    down_quantile: float = 0.33,
) -> pd.Series:
    """
    根据基准日收益分位数划分市场状态：
    - bear: r <= q_down
    - bull: r >= q_up
    - sideways: 其他

    返回：pd.Series(index=日期, value in {"bull","bear","sideways"})
    """
    r = benchmark_ret.dropna()
    q_down = r.quantile(down_quantile)
    q_up = r.quantile(up_quantile)

    regime = pd.Series(index=r.index, dtype="object")

    regime[r <= q_down] = "bear"
    regime[r >= q_up] = "bull"
    regime[(r > q_down) & (r < q_up)] = "sideways"

    return regime


def regime_performance(
    strat_ret: pd.Series,
    regime: pd.Series,
    periods_per_year: int = 252,
) -> Dict[RegimeLabel, RegimePerf]:
    """
    给定：策略收益 & 每日市场状态标签 → 统计每个状态下的表现。
    """
    strat_ret, regime = align_panels([strat_ret.to_frame("r"), regime.to_frame("regime")])
    strat_ret = strat_ret[0]["r"]
    regime = regime[1]["regime"].astype("object")

    out: Dict[RegimeLabel, RegimePerf] = {}

    for lab in ["bull", "bear", "sideways"]:
        mask = (regime == lab)
        r_sub = strat_ret[mask].dropna()
        if r_sub.empty:
            out[lab] = RegimePerf(
                regime=lab, ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan, sample_size=0
            )
            continue

        ar = ann_return(r_sub, periods_per_year)
        av = ann_vol(r_sub, periods_per_year)
        sr = sharpe_ratio(r_sub, periods_per_year=periods_per_year)
        out[lab] = RegimePerf(
            regime=lab,
            ann_return=ar,
            ann_vol=av,
            sharpe=sr,
            sample_size=len(r_sub),
        )

    return out
