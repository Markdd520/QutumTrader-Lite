# qtlite/combine/pca_combine.py
# -*- coding: utf-8 -*-
"""
PCA-based factor combination:
- 对因子做截面标准化后，在时间维度上做 PCA
- 取第一主成分作为 composite factor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, List

import numpy as np
import pandas as pd

from qtlite.core.interfaces import FactorCombiner
from qtlite.core.registry import combiner_registry
from qtlite.core.utils import align_panels, zscore


DataPanel = pd.DataFrame


@dataclass
class PCAConfig:
    """
    PCA 合成配置：
    - n_components: 使用前多少个主成分（目前实现只用前1个）
    """
    n_components: int = 1


@combiner_registry.register(category="pca", desc="PCA-based composite factor")
class PCACombiner(FactorCombiner):
    """
    PCA 合成因子：
    1. 取一组因子 panels，按日期/股票对齐
    2. 在每个日期，对各因子做截面标准化
    3. 在时间维度上，将因子视作多维特征，做 PCA
    4. 用第一主成分对应的权重，对因子做线性组合
    """

    def __init__(
        self,
        cfg: Optional[PCAConfig] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "PCACombiner")
        self.cfg = cfg or PCAConfig()
        self._weights: Optional[np.ndarray] = None
        self._factor_names: Optional[List[str]] = None

    def combine(self, factors: Mapping[str, DataPanel]) -> DataPanel:
        if not factors:
            raise ValueError("PCACombiner: empty factor dict")

        names = list(factors.keys())
        panels = list(factors.values())
        aligned = align_panels(panels)
        aligned_dict = dict(zip(names, aligned))

        # 对每个因子做截面 z-score
        z_dict = {n: zscore(df) for n, df in aligned_dict.items()}

        # 训练权重：把因子在时间维度上展开成 (T*N) × K 矩阵
        # 每个观测 = 某日某股票的一组因子值
        # 注意：只用非 NaN 行
        stacked_list = []
        for n in names:
            stacked_list.append(z_dict[n].stack())  # index: (date, asset)
        mat = pd.concat(stacked_list, axis=1)
        mat.columns = names
        mat = mat.dropna()

        X = mat.to_numpy(dtype=float)  # shape: (M, K)

        if X.shape[0] < len(names):
            # 数据太少，退化为等权
            w = np.ones(len(names)) / len(names)
        else:
            # PCA via SVD on covariance matrix
            # cov = X^T X / (M-1)
            cov = np.cov(X, rowvar=False)
            # eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            # 排序：从大到小
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]
            # 第一主成分的 eigenvector
            w = eigvecs[:, 0]
            # 归一化，使 sum|w|=1
            s = np.sum(np.abs(w))
            if s > 0:
                w = w / s

        self._weights = w
        self._factor_names = names

        # 用权重合成 composite factor
        comp = None
        for wi, (name, df) in zip(w, aligned_dict.items()):
            if wi == 0:
                continue
            if comp is None:
                comp = wi * df
            else:
                comp = comp + wi * df

        if comp is None:
            comp = next(iter(aligned_dict.values())).copy()
            comp[:] = 0.0

        return comp
