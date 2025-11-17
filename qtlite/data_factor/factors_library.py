# qtlite/data_factor/factors_library.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd

from qtlite.core.interfaces import Factor
from qtlite.core.registry import factor_registry
from .factors_base import (
    cs_rank, cs_zscore, cs_winsor,
    ts_mean, ts_std, ts_rank, delta,
)


# ========= 示例因子 1：涨跌幅波动 =========

@factor_registry.register(category="volatility", desc="past 10d return std")
class RetStd10(Factor):
    def compute(self, data):
        close = data["prices"]
        ret = close.pct_change()
        return ts_std(ret, 10)


# ========= 示例因子 2：成交量动量 =========

@factor_registry.register(category="volume", desc="volume delta 5")
class VolumeDelta5(Factor):
    def compute(self, data):
        vol = data["volume"]
        return delta(vol, 5)


# ========= 示例因子 3：VWAP - Close 价差秩 =========

@factor_registry.register(category="microstructure", desc="rank(vwap-close)")
class VwapCloseRank(Factor):
    def compute(self, data):
        vwap = data["vwap"]
        close = data["prices"]
        spr = vwap - close
        return cs_rank(spr)


# ========= 示例因子 4：价差波动（对标你现在的 VC 家族）=========

@factor_registry.register(category="volatility", desc="std(rank(vwap-close), 10)")
class SprRankStd10(Factor):
    def compute(self, data):
        vwap = data["vwap"]
        close = data["prices"]
        spr_rank = cs_rank(vwap - close)
        return ts_std(spr_rank, 10)


# ========= 示例因子 5：反转 =========

@factor_registry.register(category="reversal", desc="1-day reversal")
class Reversal1D(Factor):
    def compute(self, data):
        close = data["prices"]
        ret = close.pct_change()
        return -ret  # 简单反转


# ========= 示例因子 6：短期动量 =========

@factor_registry.register(category="momentum", desc="5-day momentum")
class Mom5(Factor):
    def compute(self, data):
        close = data["prices"]
        return close.pct_change(5)


# ========= 示例因子 7：波动率偏斜 =========

@factor_registry.register(category="volatility", desc="skewness of returns (ts 20)")
class RetSkew20(Factor):
    def compute(self, data):
        close = data["prices"]
        ret = close.pct_change().fillna(0)
        return ret.rolling(20).skew()


# ========= 示例因子 8：成交量强度 =========

@factor_registry.register(category="volume", desc="zscore(volume / mean20)")
class VolumeIntensity(Factor):
    def compute(self, data):
        vol = data["volume"]
        mean20 = ts_mean(vol, 20)
        x = vol / mean20.replace(0, pd.NA)
        return cs_zscore(x)


# ========= 示例因子 9：价差动量 =========

@factor_registry.register(category="microstructure", desc="delta(rank(vwap-close), 3)")
class SprRankDelta3(Factor):
    def compute(self, data):
        vwap = data["vwap"]
        close = data["prices"]
        r = cs_rank(vwap - close)
        return delta(r, 3)


# ========= 示例因子 10：超简单 size 因子 =========

@factor_registry.register(category="size", desc="negative log(price)")
class SizeLog(Factor):
    def compute(self, data):
        close = data["prices"]
        return -close.applymap(lambda x: np.log(max(x, 1e-6)))
