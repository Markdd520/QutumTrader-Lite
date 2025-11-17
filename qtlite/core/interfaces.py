# qtlite/core/interfaces.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Mapping, Optional, Protocol

import pandas as pd

from .enums import Frequency, SignalType, PositionSide


DataPanel = pd.DataFrame  # date × asset wide panel
SeriesLike = pd.Series


# ========= Backtest results =========

@dataclass
class BacktestResult:
    """
    Standardized backtest output.

    Attributes
    ----------
    nav : pd.Series
        Strategy equity curve (indexed by date).
    returns : pd.Series
        Strategy daily returns.
    turnover : Optional[pd.Series]
        Turnover series (if available).
    metrics : Dict[str, float]
        Summary statistics (Sharpe, max_dd, IR, etc.).
    extra : Dict[str, Any]
        Any extra info (factor IC series, exposures, etc.).
    """
    nav: SeriesLike
    returns: SeriesLike
    turnover: Optional[SeriesLike]
    metrics: Dict[str, float]
    extra: Dict[str, Any]


# ========= Factor interface =========

class Factor(ABC):
    """
    Abstract base class for cross-sectional factors.

    A Factor consumes basic market data (prices, volume, etc.)
    and produces a factor panel: date × asset.
    """

    name: str

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, data: Mapping[str, DataPanel]) -> DataPanel:
        """
        Compute factor values.

        Parameters
        ----------
        data : Mapping[str, DataPanel]
            Dict of basic inputs, e.g. {'close': df_close, 'amt': df_amt}.

        Returns
        -------
        DataPanel
            Factor panel (date × asset).
        """
        raise NotImplementedError


# ========= Factor combiner interface =========

class FactorCombiner(ABC):
    """
    Combine multiple factors into a composite factor.

    Typical implementations:
    - linear combination (static / rolling weights)
    - GA / Bayesian optimized weights
    - PCA / SVD based composite
    """

    name: str

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def combine(self, factors: Mapping[str, DataPanel]) -> DataPanel:
        """
        Combine factor panels into a single composite factor.

        Parameters
        ----------
        factors : Mapping[str, DataPanel]
            Dict: factor_name -> factor_panel.

        Returns
        -------
        DataPanel
            Composite factor panel.
        """
        raise NotImplementedError


# ========= Strategy interface =========

@dataclass
class StrategyContext:
    """
    Context info passed into strategies.
    可以放频率、资金规模、benchmark 等。
    """
    frequency: Frequency = Frequency.DAILY
    side: PositionSide = PositionSide.LONG_SHORT
    benchmark: Optional[SeriesLike] = None
    extra: Dict[str, Any] = None


class Strategy(ABC):
    """
    Strategy consumes factors (or composite factor) and
    produces portfolio signals / target weights.
    """

    name: str
    signal_type: SignalType

    def __init__(self, name: Optional[str] = None,
                 signal_type: SignalType = SignalType.WEIGHT):
        self.name = name or self.__class__.__name__
        self.signal_type = signal_type

    @abstractmethod
    def generate_signals(
        self,
        factors: Mapping[str, DataPanel],
        context: StrategyContext,
    ) -> DataPanel:
        """
        Generate signals or target weights.

        Parameters
        ----------
        factors : Mapping[str, DataPanel]
            Dict of factor panels used by the strategy.
        context : StrategyContext
            Global context (frequency, side, benchmark, etc.).

        Returns
        -------
        DataPanel
            date × asset matrix of signals or target weights.
        """
        raise NotImplementedError


# ========= Risk model & constraints =========

class RiskModel(ABC):
    """
    Risk model estimates risk quantities:
    covariance matrix, factor exposures, etc.
    """

    name: str

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def estimate_risk(
        self,
        returns: DataPanel,
        **kwargs: Any,
    ) -> Any:
        """
        Estimate risk objects (cov matrix, vol, etc.).

        Returns
        -------
        Any
            User-defined risk representation, e.g. covariance matrix.
        """
        raise NotImplementedError


class RiskConstraint(ABC):
    """
    Risk constraint adjusts target weights based on
    risk model outputs or simple rules (position limits,
    sector caps, leverage limits, turnover constraints, etc.).
    """

    name: str

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def apply(
        self,
        target_weights: DataPanel,
        risk_state: Any,
        **kwargs: Any,
    ) -> DataPanel:
        """
        Apply risk constraints to target weights.

        Parameters
        ----------
        target_weights : DataPanel
            Original weights from strategy.
        risk_state : Any
            Output of a RiskModel or other risk info.

        Returns
        -------
        DataPanel
            Adjusted weights after applying risk constraints.
        """
        raise NotImplementedError


# ========= Backtest engine protocol =========

class BacktestEngine(Protocol):
    """
    Protocol for backtest engines.

    Any concrete engine only needs to implement `run` with this signature.
    """

    def run(
        self,
        prices: DataPanel,
        strategy: Strategy,
        risk_model: Optional[RiskModel],
        constraints: Optional[Mapping[str, RiskConstraint]],
        factors: Mapping[str, DataPanel],
        context: StrategyContext,
    ) -> BacktestResult:
        ...
