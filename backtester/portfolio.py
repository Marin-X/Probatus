"""Portfolio: cash, positions, P&L. Sizes signals into orders.

Three responsibilities:
    1. Convert SignalEvents into sized OrderEvents (on_signal).
    2. Update internal state when fills come back (on_fill).
    3. Mark to market on each bar and snapshot equity (update_timeindex).

The bar-by-bar equity history is stored in ``self.history`` and exposed
as a DataFrame via ``equity_curve()`` for downstream analytics. The
fill stream is stored in ``self.fills`` so trade-level stats can be
reconstructed without a second pass through the engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .data import DataHandler
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent


@dataclass
class PortfolioSnapshot:
    """One row of the equity curve."""
    timestamp: pd.Timestamp
    cash: float
    positions_value: float
    total_equity: float
    holdings: dict[str, int]


class Portfolio(ABC):
    @abstractmethod
    def on_signal(self, event: SignalEvent) -> Iterable[OrderEvent]: ...

    @abstractmethod
    def on_fill(self, event: FillEvent) -> None: ...

    @abstractmethod
    def update_timeindex(self, event: MarketEvent) -> None: ...


class FixedFractionPortfolio(Portfolio):
    """Naive but useful baseline: each LONG signal allocates a fixed
    fraction of *current* equity to the symbol. EXIT flattens the position.

    Long-only in M1; SHORT signals are ignored.
    """

    def __init__(self, data: DataHandler, initial_cash: float = 100_000,
                 allocation_fraction: float = 0.20):
        if not 0 < allocation_fraction <= 1:
            raise ValueError("allocation_fraction must be in (0, 1]")
        self.data = data
        self.initial_cash = float(initial_cash)
        self.allocation_fraction = float(allocation_fraction)

        self.cash: float = self.initial_cash
        self.positions: dict[str, int] = defaultdict(int)
        self.history: list[PortfolioSnapshot] = []
        self.fills: list[FillEvent] = []

    # --- event handlers ------------------------------------------------------

    def on_signal(self, event: SignalEvent) -> Iterable[OrderEvent]:
        sym = event.symbol
        price = self.data.get_latest_price(sym)
        if not _is_finite_positive(price):
            return []

        if event.direction == "LONG":
            if self.positions[sym] > 0:
                return []
            target_dollars = self._current_equity() * self.allocation_fraction
            qty = int(target_dollars // price)
            if qty <= 0:
                return []
            return [OrderEvent(
                timestamp=event.timestamp, symbol=sym,
                order_type="MKT", quantity=qty, direction="BUY",
            )]

        if event.direction == "EXIT":
            qty = self.positions[sym]
            if qty <= 0:
                return []
            return [OrderEvent(
                timestamp=event.timestamp, symbol=sym,
                order_type="MKT", quantity=qty, direction="SELL",
            )]

        return []

    def on_fill(self, event: FillEvent) -> None:
        self.fills.append(event)
        sign = 1 if event.direction == "BUY" else -1
        self.positions[event.symbol] += sign * event.quantity
        self.cash -= sign * event.quantity * event.fill_price
        self.cash -= event.commission

    def update_timeindex(self, event: MarketEvent) -> None:
        positions_value = self._positions_value()
        self.history.append(PortfolioSnapshot(
            timestamp=event.timestamp,
            cash=self.cash,
            positions_value=positions_value,
            total_equity=self.cash + positions_value,
            holdings=dict(self.positions),
        ))

    # --- analytics -----------------------------------------------------------

    def equity_curve(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        rows = [{
            "timestamp": s.timestamp,
            "cash": s.cash,
            "positions_value": s.positions_value,
            "equity": s.total_equity,
        } for s in self.history]
        return pd.DataFrame(rows).set_index("timestamp")

    # --- internals -----------------------------------------------------------

    def _positions_value(self) -> float:
        total = 0.0
        for sym, qty in self.positions.items():
            if qty == 0:
                continue
            price = self.data.get_latest_price(sym)
            if _is_finite_positive(price):
                total += qty * price
        return total

    def _current_equity(self) -> float:
        return self.cash + self._positions_value()


def _is_finite_positive(x: float) -> bool:
    return x == x and x > 0