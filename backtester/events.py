"""Event types passed between components.

Event-driven design: every component emits events that others react to.
Events are immutable (frozen dataclasses) so a component can never
accidentally mutate state belonging to another component.

Per-bar flow:
    MarketEvent  -> Strategy
    SignalEvent  -> Portfolio
    OrderEvent   -> ExecutionHandler
    FillEvent    -> Portfolio
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional


@dataclass(frozen=True)
class MarketEvent:
    """Emitted by the DataHandler when a new bar arrives."""
    timestamp: datetime
    type: Literal["MARKET"] = "MARKET"


@dataclass(frozen=True)
class SignalEvent:
    """Emitted by a Strategy: 'I want exposure here.' Sizing is the
    Portfolio's job — strategies do not know cash or position sizes."""
    timestamp: datetime
    symbol: str
    direction: Literal["LONG", "SHORT", "EXIT"]
    strength: float = 1.0  # confidence weight, used optionally for sizing
    type: Literal["SIGNAL"] = "SIGNAL"


@dataclass(frozen=True)
class OrderEvent:
    """Emitted by the Portfolio after sizing a signal into a concrete order."""
    timestamp: datetime
    symbol: str
    order_type: Literal["MKT", "LMT"]
    quantity: int
    direction: Literal["BUY", "SELL"]
    limit_price: Optional[float] = None
    type: Literal["ORDER"] = "ORDER"


@dataclass(frozen=True)
class FillEvent:
    """Emitted by the ExecutionHandler when an order is filled."""
    timestamp: datetime
    symbol: str
    quantity: int
    direction: Literal["BUY", "SELL"]
    fill_price: float
    commission: float = 0.0   # M2 will populate
    slippage: float = 0.0     # M2 will populate
    type: Literal["FILL"] = "FILL"