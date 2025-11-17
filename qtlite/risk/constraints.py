# qtlite/risk/constraints.py
# -*- coding: utf-8 -*-
"""
Risk constraints:
- position limit
- sector limit
- volatility target
- turnover control

RiskModel 输出的 risk_state 在这里约定为：
- 对于协方差模型：risk_state = cov_matrix (asset × asset)
- 其他：可以是任意字典，约束内部自行约定使用方式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from qtlite.core.interfaces import RiskConstraint
from qtlite.core.registry import risk_registry
from .exposure import sector_exposure


DataPanel = pd.DataFrame


# ========= 单票限制 =========

@dataclass
class PositionLimitConfig:
    """
    单票权重上限：
    - max_abs_weight: |w_i| <= max_abs_weight
    """
    max_abs_weight: float = 0.05


@risk_registry.register(category="constraint", desc="Per-asset position limit")
class PositionLimitConstraint(RiskConstraint):
    def __init__(self, cfg: Optional[PositionLimitConfig] = None, name: Optional[str] = None):
        super().__init__(name=name or "PositionLimitConstraint")
        self.cfg = cfg or PositionLimitConfig()

    def apply(
        self,
        target_weights: DataPanel,
        risk_state: Any,
        **kwargs: Any,
    ) -> DataPanel:
        w = target_weights.copy()
        w = w.clip(lower=-self.cfg.max_abs_weight, upper=self.cfg.max_abs_weight)
        return w


# ========= 行业权重限制 =========

@dataclass
class SectorLimitConfig:
    """
    行业权重限制：
    - sector_map: 股票 -> 行业
    - max_long: 行业多头权重上限
    - min_short: 行业空头权重下限（通常为负数，例如 -0.10）
    """
    sector_map: pd.Series | None = None
    max_long: float = 0.20
    min_short: float = -0.20


@risk_registry.register(category="constraint", desc="Sector exposure limit")
class SectorLimitConstraint(RiskConstraint):
    def __init__(self, cfg: Optional[SectorLimitConfig] = None, name: Optional[str] = None):
        super().__init__(name=name or "SectorLimitConstraint")
        self.cfg = cfg or SectorLimitConfig()

    def apply(
        self,
        target_weights: DataPanel,
        risk_state: Any,
        **kwargs: Any,
    ) -> DataPanel:
        if self.cfg.sector_map is None:
            # 没有行业信息就直接返回
            return target_weights

        w = target_weights.copy()
        # 按日处理
        expo = sector_exposure(w, self.cfg.sector_map)

        # 简单做法：若某行业超限，按比例缩放该行业权重
        for dt in w.index:
            sec_w = expo.loc[dt]  # sector -> weight
            if sec_w.isna().all():
                continue
            for sec, val in sec_w.items():
                if pd.isna(val):
                    continue
                # 多头超出 max_long
                if val > self.cfg.max_long:
                    # scale因子：max_long / val
                    scale = self.cfg.max_long / val if val != 0 else 1.0
                    mask = (self.cfg.sector_map == sec)
                    w.loc[dt, mask] *= scale
                # 空头超出 min_short
                if val < self.cfg.min_short:
                    scale = self.cfg.min_short / val if val != 0 else 1.0
                    mask = (self.cfg.sector_map == sec)
                    w.loc[dt, mask] *= scale

        return w


# ========= 波动率目标 (Vol Target) =========

@dataclass
class VolTargetConfig:
    """
    波动率目标：
    - target_vol: 目标年化波动率
    - periods_per_year: 年化基数（每日 252）
    """
    target_vol: float = 0.10
    periods_per_year: int = 252


@risk_registry.register(category="constraint", desc="Volatility targeting")
class VolTargetConstraint(RiskConstraint):
    """
    基于协方差矩阵调节整体杠杆，使组合年化波动 ≈ target_vol。
    要求 risk_state 为协方差矩阵 (asset × asset)。
    """

    def __init__(self, cfg: Optional[VolTargetConfig] = None, name: Optional[str] = None):
        super().__init__(name=name or "VolTargetConstraint")
        self.cfg = cfg or VolTargetConfig()

    def apply(
        self,
        target_weights: DataPanel,
        risk_state: Any,
        **kwargs: Any,
    ) -> DataPanel:
        if risk_state is None or not isinstance(risk_state, pd.DataFrame):
            # 未提供协方差矩阵就不做调整
            return target_weights

        cov = risk_state  # asset × asset

        w = target_weights.copy()

        # 只支持协方差矩阵资产集合的子集
        common_assets = w.columns.intersection(cov.index)
        cov_sub = cov.loc[common_assets, common_assets]

        for dt in w.index:
            w_t = w.loc[dt, common_assets].values.reshape(-1, 1)  # (N,1)
            vol_daily_sq = float(w_t.T @ cov_sub.to_numpy(dtype=float) @ w_t)
            if vol_daily_sq <= 0:
                continue
            vol_annual = np.sqrt(vol_daily_sq) * np.sqrt(self.cfg.periods_per_year)
            if vol_annual == 0 or np.isnan(vol_annual):
                continue

            scale = self.cfg.target_vol / vol_annual
            w.loc[dt, common_assets] = w.loc[dt, common_assets] * scale

        return w


# ========= 换手率控制 =========

@dataclass
class TurnoverLimitConfig:
    """
    换手率限制（软约束）：
    - max_turnover: 每日最大换手率（0~1）
    """
    max_turnover: float = 0.3


@risk_registry.register(category="constraint", desc="Turnover soft limit")
class TurnoverLimitConstraint(RiskConstraint):
    """
    软约束：若某日目标换手率 > max_turnover，则按比例缩小调仓幅度。
    约定 risk_state 可以传入 prev_weights（上一日权重）：
        risk_state = {"prev_weights": prev_weights_df}
    若未传，则不生效。
    """

    def __init__(self, cfg: Optional[TurnoverLimitConfig] = None, name: Optional[str] = None):
        super().__init__(name=name or "TurnoverLimitConstraint")
        self.cfg = cfg or TurnoverLimitConfig()

    def apply(
        self,
        target_weights: DataPanel,
        risk_state: Any,
        **kwargs: Any,
    ) -> DataPanel:
        if not isinstance(risk_state, dict) or "prev_weights" not in risk_state:
            # 没有 prev_weights 就不做约束
            return target_weights

        prev_w: DataPanel = risk_state["prev_weights"]
        w, prev = target_weights.align(prev_w, join="left", axis=0)
        prev = prev.fillna(0.0)

        out = w.copy()
        for dt in w.index:
            w_t = w.loc[dt]
            prev_t = prev.loc[dt]
            turnover = 0.5 * (w_t - prev_t).abs().sum()
            if turnover <= self.cfg.max_turnover or turnover == 0:
                continue
            # 缩放调仓幅度
            scale = self.cfg.max_turnover / turnover
            out.loc[dt] = prev_t + scale * (w_t - prev_t)

        return out
