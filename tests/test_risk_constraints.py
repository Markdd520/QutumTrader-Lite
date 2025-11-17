# tests/test_risk_constraints.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from qtlite.risk.constraints import (
    PositionLimitConfig,
    PositionLimitConstraint,
    SectorLimitConfig,
    SectorLimitConstraint,
    VolTargetConfig,
    VolTargetConstraint,
)
from qtlite.risk.cov_models import HistoricalCovModel, HistCovConfig


def test_position_limit_basic(toy_returns):
    # 构造一组超限权重
    dates = toy_returns.index
    cols = toy_returns.columns
    w = pd.DataFrame(0.2, index=dates, columns=cols)  # 20%

    cons = PositionLimitConstraint(
        PositionLimitConfig(max_abs_weight=0.05)
    )
    w2 = cons.apply(w, risk_state=None)

    assert (w2.abs() <= 0.0500001).all().all()


def test_sector_limit_shrinks_overweight(toy_returns, toy_industry):
    # 单日权重：Sec1 超限，Sec2/3 正常
    cols = toy_returns.columns
    w0 = pd.Series(
        data=[0.3, 0.3, 0.1, 0.1, 0.2],  # Sec1: 0.6, Sec2:0.2, Sec3:0.2
        index=cols,
        name=toy_returns.index[0],
    )
    w = w0.to_frame().T  # 1×N

    cons = SectorLimitConstraint(
        SectorLimitConfig(
            sector_map=toy_industry,
            max_long=0.4,     # Sec1 超过 0.4
            min_short=-0.4,
        )
    )

    w2 = cons.apply(w, risk_state=None)
    # 新的 Sec1 权重应该 <= 0.4
    sec1_mask = (toy_industry == "Sec1")
    sec1_weight_new = w2.loc[w2.index[0], sec1_mask].sum()
    assert sec1_weight_new <= 0.400001


def test_vol_target_moves_towards_target(toy_returns):
    # 用 HistoricalCovModel 估一个协方差
    cov_model = HistoricalCovModel(HistCovConfig(window=30, min_periods=10))
    cov = cov_model.estimate_risk(toy_returns)

    # 构造一个随机权重（单日）
    cols = toy_returns.columns
    w_series = pd.Series(
        data=[0.2, 0.2, -0.2, -0.1, -0.1],
        index=cols,
        name=toy_returns.index[0],
    )
    w = w_series.to_frame().T  # 1×N

    # 计算原始年化波动
    Sigma = cov.loc[cols, cols].to_numpy(dtype=float)
    w_vec = w_series.values.reshape(-1, 1)
    vol0 = float(np.sqrt(w_vec.T @ Sigma @ w_vec) * np.sqrt(252))

    vt = VolTargetConstraint(
        VolTargetConfig(target_vol=0.10, periods_per_year=252)
    )
    w2 = vt.apply(w, risk_state=cov)

    w2_vec = w2.iloc[0].values.reshape(-1, 1)
    vol1 = float(np.sqrt(w2_vec.T @ Sigma @ w2_vec) * np.sqrt(252))

    # 不要求完全等于，但应该比 vol0 靠近目标 0.10
    target = 0.10
    assert abs(vol1 - target) < abs(vol0 - target) or np.isclose(vol1, target)
