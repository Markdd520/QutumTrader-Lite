# qtlite/data_factor/pipeline.py
# -*- coding: utf-8 -*-
"""
Factor & data pipeline
- forward returns
- basic cleaning (winsorize / z-score)
- simple industry neutralization

这个模块的目标：
给你一个“对标私募，但结构极简”的因子预处理管线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from qtlite.core.utils import winsorize, zscore, align_panels


DataPanel = pd.DataFrame


# ========= 配置 =========

@dataclass
class FactorPipelineConfig:
    """
    因子预处理配置（可以在 configs/*.yml 中映射过来）

    Attributes
    ----------
    winsor : bool
        是否做截面去极值（按分位数裁剪）。
    winsor_lower : float
        去极值下分位数。
    winsor_upper : float
        去极值上分位数。
    standardize : bool
        是否做截面标准化（z-score）。
    neutralize : bool
        是否按行业做中性化（减去行业均值）。
    """
    winsor: bool = True
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    standardize: bool = True
    neutralize: bool = True


# ========= forward returns =========

def compute_forward_returns(
    prices: DataPanel,
    horizon: int = 1,
    log: bool = False,
) -> DataPanel:
    """
    Compute forward returns from price panel.

    Parameters
    ----------
    prices : DataPanel
        price panel (date × asset).
    horizon : int
        forward window in days, default 1.
    log : bool
        whether to use log-returns.

    Returns
    -------
    DataPanel
        forward returns r_{t,t+h} 对齐在 t（即用来检验 t 日因子的预测力）。
    """
    prices = prices.sort_index()
    if log:
        ret = np.log(prices.shift(-horizon) / prices)
    else:
        ret = prices.shift(-horizon) / prices - 1.0
    return ret


# ========= 行业中性化（简化版：减去行业均值） =========

def neutralize_by_industry(
    factor: DataPanel,
    industry: pd.Series,
) -> DataPanel:
    """
    行业中性化（简化版）：
    对于每个交易日，将因子值减去该股票所属行业的截面均值。

    Parameters
    ----------
    factor : DataPanel
        因子宽表 (date × asset)。
    industry : pd.Series
        index = asset, value = industry_code。

    Returns
    -------
    DataPanel
        行业中性化后的因子。
    """
    # 对齐列
    common_cols = factor.columns.intersection(industry.index)
    f = factor[common_cols]
    ind = industry.loc[common_cols]

    def _neu(s: pd.Series) -> pd.Series:
        # s: 某一天所有股票的因子
        df = pd.DataFrame({"f": s, "ind": ind})
        grouped = df.groupby("ind")["f"]
        # 组内减均值
        return s - grouped.transform("mean")

    return f.apply(_neu, axis=1)


# ========= 因子预处理主入口 =========

def process_factors(
    raw_factors: Mapping[str, DataPanel],
    cfg: Optional[FactorPipelineConfig] = None,
    industry: Optional[pd.Series] = None,
) -> Dict[str, DataPanel]:
    """
    对一组因子做统一预处理：去极值 → 行业中性化 → 标准化。

    Parameters
    ----------
    raw_factors : Mapping[str, DataPanel]
        dict: name -> factor_panel (date × asset)。
    cfg : FactorPipelineConfig, optional
        预处理配置；若为 None，则用默认参数。
    industry : pd.Series, optional
        股票 → 行业 映射，用于行业中性化。

    Returns
    -------
    Dict[str, DataPanel]
        预处理后的因子字典。
    """
    if cfg is None:
        cfg = FactorPipelineConfig()

    processed: Dict[str, DataPanel] = {}

    # 统一对齐日期 × 股票
    panels = list(raw_factors.values())
    aligned_panels = align_panels(panels)
    factor_names = list(raw_factors.keys())
    aligned = dict(zip(factor_names, aligned_panels))

    for name, f in aligned.items():
        x = f.copy()

        if cfg.winsor:
            x = winsorize(x, lower=cfg.winsor_lower, upper=cfg.winsor_upper)

        if cfg.neutralize and industry is not None:
            x = neutralize_by_industry(x, industry=industry)

        if cfg.standardize:
            x = zscore(x)

        processed[name] = x

    return processed
