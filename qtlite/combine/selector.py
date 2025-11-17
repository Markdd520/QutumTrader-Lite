# qtlite/combine/selector.py
# -*- coding: utf-8 -*-
"""
Factor selector:
- 根据 IC/IR 等筛选因子子集
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping, Dict

import pandas as pd


DataPanel = pd.DataFrame


@dataclass
class SelectorConfig:
    """
    筛选规则（可以非常简单，也可以后面扩展复杂逻辑）。
    """
    min_ic_ir: float = 0.0       # 最小 IC-IR
    min_rank_ic_ir: float = 0.0  # 最小 RankIC-IR
    min_sample_size: int = 50    # 至少多少截面样本
    max_factors: int | None = None  # 最多保留多少个因子（按 rank_ic_ir 排序）


class FactorSelector:
    """
    FactorSelector:
    - 输入：factors dict + score_table（由 factor_score_table 计算）
    - 输出：筛选后的因子 dict
    """

    def __init__(self, cfg: SelectorConfig | None = None):
        self.cfg = cfg or SelectorConfig()

    def select(
        self,
        factors: Mapping[str, DataPanel],
        score_table: pd.DataFrame,
    ) -> Dict[str, DataPanel]:
        df = score_table.copy()

        # 基于 IC-IR / RankIC-IR 和样本数筛选
        mask = (
            (df["ic_ir"] >= self.cfg.min_ic_ir)
            & (df["rank_ic_ir"] >= self.cfg.min_rank_ic_ir)
            & (df["sample_size"] >= self.cfg.min_sample_size)
        )
        df = df[mask]

        # 排序：优先按 rank_ic_ir
        df = df.sort_values("rank_ic_ir", ascending=False)

        if self.cfg.max_factors is not None:
            df = df.head(self.cfg.max_factors)

        selected_names = df.index.tolist()

        out: Dict[str, DataPanel] = {}
        for name in selected_names:
            if name in factors:
                out[name] = factors[name]

        return out
