"""Transaction cost models: commission and slippage.

Two abstract bases:
    CommissionModel  - per-trade fees (per-share, percentage, fixed)
    SlippageModel    - adverse price adjustment from spread / impact

Concrete implementations are pluggable into NextBarOpenExecutor in the
next step. Defaults are zero-cost (preserves M1 behavior); production
backtests pass realistic models in.

Conventions:
    - commission() returns dollars, always >= 0
    - slippage() returns price units, signed (+ for BUY, - for SELL)
      so the executor adds it to the open price unconditionally.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .events import OrderEvent


# --- Commission models -------------------------------------------------------

class CommissionModel(ABC):
    @abstractmethod
    def commission(self, order: OrderEvent, fill_price: float) -> float:
        """Dollars charged for this fill. Always >= 0."""


class ZeroCommission(CommissionModel):
    """No commission. M1 default; useful as a baseline."""
    def commission(self, order: OrderEvent, fill_price: float) -> float:
        return 0.0


class PerShareCommission(CommissionModel):
    """IBKR-style: per-share rate, with a per-trade minimum and a
    cap as a fraction of trade value (so it doesn't blow up on penny stocks).

    Defaults match IBKR Pro's tiered schedule:
        per_share        = $0.0035
        min_per_trade    = $0.35
        max_pct_of_value = 1.0%   (cap)
    """
    def __init__(self, per_share: float = 0.0035,
                 min_per_trade: float = 0.35,
                 max_pct_of_value: float = 0.01):
        self.per_share = per_share
        self.min_per_trade = min_per_trade
        self.max_pct_of_value = max_pct_of_value

    def commission(self, order: OrderEvent, fill_price: float) -> float:
        raw = order.quantity * self.per_share
        capped = min(raw, order.quantity * fill_price * self.max_pct_of_value)
        return max(capped, self.min_per_trade)


class PercentCommission(CommissionModel):
    """Percentage of trade value. Common in crypto.

    Default 0.1% matches Coinbase Advanced taker fee for most retail tiers.
    """
    def __init__(self, rate: float = 0.001):
        self.rate = rate

    def commission(self, order: OrderEvent, fill_price: float) -> float:
        return order.quantity * fill_price * self.rate


# --- Slippage models ---------------------------------------------------------

class SlippageModel(ABC):
    @abstractmethod
    def slippage(self, order: OrderEvent, bar: pd.Series) -> float:
        """Adverse price adjustment in price units.

        BUY orders: positive (you pay above the open).
        SELL orders: negative (you receive below the open).
        Executor adds this to the open price to get fill price.
        """


class NoSlippage(SlippageModel):
    """No slippage. M1 default."""
    def slippage(self, order: OrderEvent, bar: pd.Series) -> float:
        return 0.0


class HalfSpreadSlippage(SlippageModel):
    """Pay half the bid-ask spread on each fill.

    Defaults to 2 bps half-spread (= 4 bps full spread), typical for
    liquid US large-cap names. Increase for less liquid stocks.
    """
    def __init__(self, half_spread_bps: float = 2.0):
        self.half_spread_frac = half_spread_bps / 10_000.0

    def slippage(self, order: OrderEvent, bar: pd.Series) -> float:
        sign = 1.0 if order.direction == "BUY" else -1.0
        return sign * float(bar["Open"]) * self.half_spread_frac

class AlmgrenChrissImpact(SlippageModel):
    """Square-root market impact model.

    cost_per_share = eta * sigma * price * sqrt(quantity / volume)

    Adverse price move from consuming liquidity. Scales with the
    square root of participation rate (Q/V), which matches empirical
    estimates from Almgren-Thum-Hauptmann-Li (2005).

    Defaults:
        eta   = 0.1    impact coefficient (US equities, ~empirical)
        sigma = 0.02   daily vol (typical liquid large-cap)

    For participation < 0.1% (retail-sized orders on liquid names) this
    is essentially zero — model is mainly relevant for institutional
    flow, multi-percent-ADV trades, or illiquid names. It's here so
    you can plug high-volume strategies in later and have the costs
    show up correctly.
    """
    def __init__(self, eta: float = 0.1, sigma: float = 0.02):
        self.eta = eta
        self.sigma = sigma

    def slippage(self, order: OrderEvent, bar: pd.Series) -> float:
        volume = float(bar.get("Volume", 0.0))
        if volume <= 0:
            return 0.0
        participation = order.quantity / volume
        price = float(bar["Open"])
        impact = self.eta * self.sigma * price * (participation ** 0.5)
        sign = 1.0 if order.direction == "BUY" else -1.0
        return sign * impact