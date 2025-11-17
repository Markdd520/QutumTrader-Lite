# qtlite/combine/scoring.py
# -*- coding: utf-8 -*-
"""
Factor scoring utilities:
- IC / RankIC 时间序列
- 平均 IC, IC-IR 等
"""

from __future__ import annotations

from typing import Mapping, Dict

import numpy as np
import pandas as pd

from qtlite.backtest.performance import factor_ic_series
from qtlite.core.utils import align_panels


DataPanel = pd.DataFrame


def factor_score_table(
    factors: Mapping[str, DataPanel],
    fwd_ret: DataPanel,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    对一组因子做 IC 相关打分。

    返回列：
    - ic_mean, ic_std, ic_ir
    - rank_ic_mean, rank_ic_std, rank_ic_ir
    - sample_size
    """
    names = list(factors.keys())
    scores = []

    for name, panel in factors.items():
        f, r = align_panels([panel, fwd_ret])
        f = f[0]
        r = r[1]

        ic_s, ric_s = factor_ic_series(f, r)
        ic_mu = float(ic_s.mean())
        ic_sd = float(ic_s.std(ddof=1))
        ric_mu = float(ric_s.mean())
        ric_sd = float(ric_s.std(ddof=1))

        ic_ir = ic_mu / ic_sd if ic_sd > 0 else np.nan
        ric_ir = ric_mu / ric_sd if ric_sd > 0 else np.nan

        scores.append(
            dict(
                factor=name,
                ic_mean=ic_mu,
                ic_std=ic_sd,
                ic_ir=ic_ir,
                rank_ic_mean=ric_mu,
                rank_ic_std=ric_sd,
                rank_ic_ir=ric_ir,
                sample_size=int(ic_s.notna().sum()),
            )
        )

    df = pd.DataFrame(scores).set_index("factor")
    return df
