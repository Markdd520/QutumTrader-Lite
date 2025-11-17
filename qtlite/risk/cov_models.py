# qtlite/risk/cov_models.py
# -*- coding: utf-8 -*-
"""
Covariance models:
- Historical covariance
- EWMA covariance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from qtlite.core.interfaces import RiskModel
from qtlite.core.registry import risk_registry
from qtlite.core.utils import align_panels


DataPanel = pd.DataFrame


@dataclass
class HistCovConfig:
    """
    历史协方差配置：
    - window: 使用最近多少期收益
    - min_periods: 至少多少期才计算
    """
    window: int = 252
    min_periods: int = 60


@risk_registry.register(category="cov", desc="Rolling historical covariance")
class HistoricalCovModel(RiskModel):
    """
    简单历史协方差模型：
    - 输出：资产协方差矩阵（DataFrame）
    - 注意：此处实现为“全样本一次性估计”版本，
      上层可以在滚动回测中自己控制训练窗口。
    """

    def __init__(self, cfg: Optional[HistCovConfig] = None, name: Optional[str] = None):
        super().__init__(name=name or "HistoricalCovModel")
        self.cfg = cfg or HistCovConfig()

    def estimate_risk(
        self,
        returns: DataPanel,
        **kwargs: Any,
    ) -> DataPanel:
        """
        使用末尾 window 期收益估计协方差。
        """
        if returns.empty:
            return pd.DataFrame()

        r = returns.tail(self.cfg.window)
        if r.shape[0] < self.cfg.min_periods:
            # 数据太少，返回空
            return pd.DataFrame(index=r.columns, columns=r.columns)

        cov = r.cov()
        return cov


@dataclass
class EWMACovConfig:
    """
    EWMA 协方差配置：
    - lambda_: 衰减因子（越接近 1 越长记忆）
    - min_periods: 最少样本数
    """
    lambda_: float = 0.94
    min_periods: int = 60


@risk_registry.register(category="cov", desc="EWMA covariance model")
class EWMACovModel(RiskModel):
    """
    EWMA 协方差模型：
    - 对每个资产收益做指数加权
    - 最终返回协方差矩阵
    """

    def __init__(self, cfg: Optional[EWMACovConfig] = None, name: Optional[str] = None):
        super().__init__(name=name or "EWMACovModel")
        self.cfg = cfg or EWMACovConfig()

    def estimate_risk(
        self,
        returns: DataPanel,
        **kwargs: Any,
    ) -> DataPanel:
        if returns.empty:
            return pd.DataFrame()

        r = returns.dropna(how="all")
        if r.shape[0] < self.cfg.min_periods:
            return pd.DataFrame(index=r.columns, columns=r.columns)

        lam = self.cfg.lambda_
        # 权重从旧到新：w_t = lam^(T-1-t) * (1-lam)
        T = r.shape[0]
        idx = np.arange(T)
        w = (1 - lam) * (lam ** (T - 1 - idx))
        w = w / w.sum()

        X = r.to_numpy(dtype=float)  # T×N
        X = X - np.average(X, axis=0, weights=w).reshape(1, -1)

        # 加权协方差：Σ = Xᵀ diag(w) X
        W = np.diag(w)
        cov = X.T @ W @ X

        cov_df = pd.DataFrame(cov, index=r.columns, columns=r.columns)
        return cov_df
