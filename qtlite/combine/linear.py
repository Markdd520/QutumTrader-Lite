# qtlite/combine/linear.py
# -*- coding: utf-8 -*-
"""
Linear factor combiners:
- Equal weight
- User-specified weights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Dict, Optional

import numpy as np
import pandas as pd

from qtlite.core.interfaces import FactorCombiner
from qtlite.core.registry import combiner_registry
from qtlite.core.utils import align_panels


DataPanel = pd.DataFrame


@dataclass
class LinearWeightsConfig:
    """
    线性权重配置：
    - weights: dict[name -> weight] 或 None（等权）
    - normalize: 是否在内部自动归一化（sum|w|=1）
    """
    weights: Optional[Dict[str, float]] = None
    normalize: bool = True


@combiner_registry.register(category="linear", desc="User-specified linear combination")
class LinearCombiner(FactorCombiner):
    """
    线性组合：
    - 支持给定因子权重
    - 默认会对权重做归一化处理
    """

    def __init__(
        self,
        cfg: Optional[LinearWeightsConfig] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "LinearCombiner")
        self.cfg = cfg or LinearWeightsConfig()

    def combine(self, factors: Mapping[str, DataPanel]) -> DataPanel:
        if not factors:
            raise ValueError("LinearCombiner: empty factor dict")

        names = list(factors.keys())
        panels = list(factors.values())

        # 对齐所有因子
        aligned = align_panels(panels)
        aligned_dict = dict(zip(names, aligned))

        # 权重
        if self.cfg.weights is None:
            # 等权
            w = np.ones(len(names), dtype=float)
        else:
            w = np.array([self.cfg.weights.get(n, 0.0) for n in names], dtype=float)

        if self.cfg.normalize:
            s = np.sum(np.abs(w))
            if s > 0:
                w = w / s

        # 逐日线性组合
        # composite = sum_i w_i * factor_i
        comp = None
        for wi, (name, panel) in zip(w, aligned_dict.items()):
            if wi == 0:
                continue
            if comp is None:
                comp = wi * panel
            else:
                comp = comp + wi * panel

        if comp is None:
            # 所有权重为0
            comp = next(iter(aligned_dict.values())).copy()
            comp[:] = 0.0

        return comp


@combiner_registry.register(category="linear", desc="Equal weight combination")
class EqualWeightCombiner(FactorCombiner):
    """
    等权合成（所有因子权重相等，sum|w|=1）
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name or "EqualWeightCombiner")

    def combine(self, factors: Mapping[str, DataPanel]) -> DataPanel:
        if not factors:
            raise ValueError("EqualWeightCombiner: empty factor dict")

        names = list(factors.keys())
        cfg = LinearWeightsConfig(weights={n: 1.0 for n in names}, normalize=True)
        lin = LinearCombiner(cfg=cfg)
        return lin.combine(factors)
