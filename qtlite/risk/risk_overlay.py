# qtlite/risk/risk_overlay.py
# -*- coding: utf-8 -*-
"""
Risk overlay helpers:
- apply multiple constraints sequentially
"""

from __future__ import annotations

from typing import Mapping, Any

import pandas as pd

from qtlite.core.interfaces import RiskConstraint


DataPanel = pd.DataFrame


def apply_constraints(
    weights: DataPanel,
    constraints: Mapping[str, RiskConstraint],
    risk_state: Any,
) -> DataPanel:
    """
    按照给定顺序依次应用多个 RiskConstraint。

    注意：
    - SimpleBacktestEngine 已经有类似逻辑；
      这个函数可用于独立研究：先生成权重，再单独叠加风控。
    """
    w = weights.copy()
    for name, cons in constraints.items():
        w = cons.apply(w, risk_state)
    return w
