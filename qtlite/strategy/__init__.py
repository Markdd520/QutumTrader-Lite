# qtlite/strategy/__init__.py
# -*- coding: utf-8 -*-

from .factor_longshort import (
    FactorLongShortConfig,
    FactorLongShortStrategy,
    FactorTopKConfig,
    FactorTopKLongStrategy,
)

from .signals import (
    cs_rank,
    cs_zscore,
    winsorize,
    neutralize,
    scale_to_gross,
    normalize_long_only,
)

from .position_sizing import (
    equal_weight,
    inv_vol_weight,
    map_groups_to_weights,
)

from .portfolio import (
    compute_turnover,
    apply_transaction_cost,
    rebalance,
)

from .execution_sim import (
    simple_slippage,
    impact_cost,
)

__all__ = [
    # 核心策略
    "FactorLongShortConfig",
    "FactorLongShortStrategy",
    "FactorTopKConfig",
    "FactorTopKLongStrategy",
    # 信号工具
    "cs_rank",
    "cs_zscore",
    "winsorize",
    "neutralize",
    "scale_to_gross",
    "normalize_long_only",
    # 仓位 / 组合
    "equal_weight",
    "inv_vol_weight",
    "map_groups_to_weights",
    "compute_turnover",
    "apply_transaction_cost",
    "rebalance",
    # 执行仿真
    "simple_slippage",
    "impact_cost",
]
