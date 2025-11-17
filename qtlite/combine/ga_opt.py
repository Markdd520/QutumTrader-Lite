# qtlite/combine/ga_opt.py
# -*- coding: utf-8 -*-
"""
GA-like factor weight search:
- 简化的 GA / 进化搜索，不依赖 deap
- 目标：最大化 composite factor 的 IC-IR 或 mean(IC)

流程：
1. 输入一组因子 panels + forward returns（在 __init__ 中给）
2. combine(factors) 时：
   - 若尚未优化，自动跑 GA，在训练窗口上找一组最优权重
   - 用最优权重合成完整历史上的 composite factor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Dict, Optional, Literal, List, Tuple

import numpy as np
import pandas as pd

from qtlite.core.interfaces import FactorCombiner
from qtlite.core.registry import combiner_registry
from qtlite.core.utils import align_panels
from qtlite.backtest.performance import factor_ic_series


DataPanel = pd.DataFrame
ObjectiveType = Literal["IC_MEAN", "IC_IR"]


@dataclass
class GAConfig:
    """
    简化 GA 配置（对标你现在 build_composite_ga_fast 里的核心参数）
    """
    pop_size: int = 80
    n_gen: int = 100
    cxpb: float = 0.8
    mutpb: float = 0.2
    train_start: Optional[pd.Timestamp] = None
    train_end: Optional[pd.Timestamp] = None
    random_state: int = 42

    # 权重范围控制（sum|w|=1 约束下，单个系数大致控制）
    w_min: float = -1.0
    w_max: float = 1.0

    # mutation 噪声
    mut_sigma: float = 0.2


@combiner_registry.register(category="ga", desc="GA-like optimized linear combination")
class GACombiner(FactorCombiner):
    """
    GA 风格优化的因子权重组合：
    - 个体 = 一组权重 w (len = n_factors)
    - 约束：sum(|w|) = 1
    - 适应度 = composite factor IC-IR 或 mean(IC)

    使用方式：
    ga = GACombiner(fwd_ret=fwd_ret, cfg=GAConfig(...), objective="IC_IR")
    composite_factor = ga.combine(factors_dict)
    """

    def __init__(
        self,
        fwd_ret: DataPanel,
        cfg: Optional[GAConfig] = None,
        objective: ObjectiveType = "IC_IR",
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "GACombiner")
        self.fwd_ret_full = fwd_ret
        self.cfg = cfg or GAConfig()
        self.objective: ObjectiveType = objective
        self._best_weights: Optional[np.ndarray] = None
        self._factor_names: Optional[List[str]] = None

        # RNG
        self._rng = np.random.default_rng(self.cfg.random_state)

    # ========= 主入口 =========

    def combine(self, factors: Mapping[str, DataPanel]) -> DataPanel:
        if not factors:
            raise ValueError("GACombiner: empty factor dict")

        names = list(factors.keys())
        panels = list(factors.values())

        # 对齐因子 & forward returns
        aligned = align_panels(panels + [self.fwd_ret_full])
        factor_panels = aligned[:-1]
        fwd_ret = aligned[-1]

        factors_aligned = dict(zip(names, factor_panels))

        # 训练窗口裁剪
        train_factors, train_fwd = self._clip_train_window(factors_aligned, fwd_ret)

        # 若还未优化，先跑 GA
        if self._best_weights is None or self._factor_names != names:
            w_best, score_best = self._run_ga(train_factors, train_fwd)
            self._best_weights = w_best
            self._factor_names = names
            # 这里不打印日志，留给上层去 log

        # 用最优权重在完整历史上合成
        comp = self._combine_with_weights(factors_aligned, self._best_weights)
        return comp

    # ========= 内部工具 =========

    def _clip_train_window(
        self,
        factors: Dict[str, DataPanel],
        fwd_ret: DataPanel,
    ) -> Tuple[Dict[str, DataPanel], DataPanel]:
        """
        根据 cfg.train_start/end 裁剪训练区间；若未指定，则用全样本。
        """
        start = self.cfg.train_start or fwd_ret.index.min()
        end = self.cfg.train_end or fwd_ret.index.max()

        fwd_ret_train = fwd_ret.loc[(fwd_ret.index >= start) & (fwd_ret.index <= end)]

        fac_train: Dict[str, DataPanel] = {}
        for name, panel in factors.items():
            fac_train[name] = panel.reindex(fwd_ret_train.index)

        return fac_train, fwd_ret_train

    def _run_ga(
        self,
        factors: Dict[str, DataPanel],
        fwd_ret: DataPanel,
    ) -> Tuple[np.ndarray, float]:
        """
        核心 GA 过程：
        - 初始化 population
        - 多代进化：选择 + 交叉 + 变异
        - 返回最优权重 & 适应度
        """
        n_factors = len(factors)
        names = list(factors.keys())
        panels = list(factors.values())
        # 对齐
        panels, fwd_ret = align_panels(panels + [fwd_ret])
        panels = panels[:-1]
        fwd_ret = fwd_ret[-1]

        # 预先把因子堆成 3D 结构：time × asset × factor
        # 方便在迭代中快速计算 composite
        idx = panels[0].index
        cols = panels[0].columns
        T = len(idx)
        N = len(cols)
        K = n_factors

        tensor = np.zeros((T, N, K), dtype=float)
        for k, df in enumerate(panels):
            df = df.reindex(index=idx, columns=cols)
            tensor[:, :, k] = df.to_numpy(dtype=float)

        fwd_arr = fwd_ret.reindex(index=idx, columns=cols).to_numpy(dtype=float)

        # mask：哪些 date × asset 有有效收益
        valid_mask = ~np.isnan(fwd_arr)

        def _eval(w: np.ndarray) -> float:
            """
            适应度函数：
            - composite = sum_k w_k * factor_k
            - 对每天的截面做 IC / RankIC（这里只用 IC）
            - objective:
                IC_MEAN: mean(IC)
                IC_IR:   mean(IC) / std(IC)
            """
            # composite: T × N
            comp = np.tensordot(tensor, w, axes=(2, 0))  # (T,N)

            ic_list: List[float] = []
            for t in range(T):
                r_t = fwd_arr[t]
                f_t = comp[t]
                mask = valid_mask[t] & np.isfinite(f_t)
                if mask.sum() < 5:
                    continue
                x = f_t[mask]
                y = r_t[mask]
                if np.all(x == x[0]) or np.all(y == y[0]):
                    continue
                ic_val = np.corrcoef(x, y)[0, 1]
                if np.isnan(ic_val):
                    continue
                ic_list.append(float(ic_val))

            if not ic_list:
                return -1e9  # 无效权重给很差的分

            ic_arr = np.asarray(ic_list, dtype=float)
            ic_mean = float(ic_arr.mean())
            ic_std = float(ic_arr.std(ddof=1)) if ic_arr.size > 1 else 0.0

            if self.objective == "IC_MEAN":
                return ic_mean
            else:  # IC_IR
                if ic_std <= 0:
                    return -1e9
                return ic_mean / ic_std

        # 个体表示：长度 K 的权重向量
        def _random_individual() -> np.ndarray:
            w = self._rng.normal(0.0, 1.0, size=K)
            w = np.clip(w, self.cfg.w_min, self.cfg.w_max)
            return self._normalize_weights(w)

        def _normalize_weights(w: np.ndarray) -> np.ndarray:
            # sum|w|=1
            s = np.sum(np.abs(w))
            if s == 0:
                # fallback：给一个小偏移，避免全0
                w = np.ones_like(w) / K
                s = np.sum(np.abs(w))
            return w / s

        def _crossover(w1: np.ndarray, w2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if K <= 1:
                return w1.copy(), w2.copy()
            point = self._rng.integers(1, K)
            c1 = np.concatenate([w1[:point], w2[point:]])
            c2 = np.concatenate([w2[:point], w1[point:]])
            return _normalize_weights(c1), _normalize_weights(c2)

        def _mutate(w: np.ndarray) -> np.ndarray:
            noise = self._rng.normal(0.0, self.cfg.mut_sigma, size=K)
            w_new = w + noise
            w_new = np.clip(w_new, self.cfg.w_min, self.cfg.w_max)
            return _normalize_weights(w_new)

        # 初始化种群
        pop = [_random_individual() for _ in range(self.cfg.pop_size)]
        fitness = [ _eval(ind) for ind in pop ]

        def _tournament_select(k: int = 2) -> int:
            idxs = self._rng.integers(0, len(pop), size=k)
            best_i = idxs[0]
            best_f = fitness[best_i]
            for i in idxs[1:]:
                if fitness[i] > best_f:
                    best_i = i
                    best_f = fitness[i]
            return best_i

        best_idx = int(np.argmax(fitness))
        best_w = pop[best_idx].copy()
        best_score = float(fitness[best_idx])

        for g in range(self.cfg.n_gen):
            new_pop: List[np.ndarray] = []
            new_fitness: List[float] = []

            # 精英保留
            new_pop.append(best_w.copy())
            new_fitness.append(best_score)

            # 其余通过 GA 操作生成
            while len(new_pop) < self.cfg.pop_size:
                # 选择
                i1 = _tournament_select()
                i2 = _tournament_select()
                p1 = pop[i1].copy()
                p2 = pop[i2].copy()

                # 交叉
                if self._rng.random() < self.cfg.cxpb:
                    c1, c2 = _crossover(p1, p2)
                else:
                    c1, c2 = p1, p2

                # 变异
                if self._rng.random() < self.cfg.mutpb:
                    c1 = _mutate(c1)
                if self._rng.random() < self.cfg.mutpb:
                    c2 = _mutate(c2)

                # 评估
                f1 = _eval(c1)
                new_pop.append(c1)
                new_fitness.append(f1)

                if len(new_pop) < self.cfg.pop_size:
                    f2 = _eval(c2)
                    new_pop.append(c2)
                    new_fitness.append(f2)

            pop = new_pop
            fitness = new_fitness

            # 更新全局最优
            gen_best_idx = int(np.argmax(fitness))
            gen_best_w = pop[gen_best_idx]
            gen_best_score = float(fitness[gen_best_idx])

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_w = gen_best_w.copy()

        return best_w, best_score

    def _combine_with_weights(
        self,
        factors: Dict[str, DataPanel],
        weights: np.ndarray,
    ) -> DataPanel:
        names = list(factors.keys())
        panels = list(factors.values())
        aligned = align_panels(panels)
        aligned_dict = dict(zip(names, aligned))

        comp = None
        for wi, (name, panel) in zip(weights, aligned_dict.items()):
            if wi == 0:
                continue
            if comp is None:
                comp = wi * panel
            else:
                comp = comp + wi * panel

        if comp is None:
            comp = next(iter(aligned_dict.values())).copy()
            comp[:] = 0.0

        return comp
