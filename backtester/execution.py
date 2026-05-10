"""Execution handlers: simulate the broker.

M2 wires in cost models — commission and slippage — through optional
constructor arguments. Defaults are zero-cost so M1 backtests still
work unchanged; production runs pass realistic models in.

Per fill:
    fill_price  = bar's open + slippage(order, bar)
    commission  = commission(order, fill_price)

Both fields land on the FillEvent and the portfolio subtracts them
from cash via on_fill.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from .costs import CommissionModel, NoSlippage, SlippageModel, ZeroCommission
from .data import DataHandler
from .events import FillEvent, MarketEvent, OrderEvent


class ExecutionHandler(ABC):
    @abstractmethod
    def queue_order(self, order: OrderEvent) -> None: ...

    @abstractmethod
    def execute_pending(self, event: MarketEvent) -> Iterable[FillEvent]: ...


class NextBarOpenExecutor(ExecutionHandler):
    """Fills queued orders at the OPEN of the next bar, optionally with
    commission and slippage applied.

    Backward-compatible with M1: if no cost models are passed, behaves
    exactly like the original (zero costs).
    """

    def __init__(self, data: DataHandler,
                 commission_model: Optional[CommissionModel] = None,
                 slippage_model: Optional[SlippageModel] = None):
        self.data = data
        self.commission_model = commission_model or ZeroCommission()
        self.slippage_model = slippage_model or NoSlippage()
        self._pending: list[OrderEvent] = []

    def queue_order(self, order: OrderEvent) -> None:
        self._pending.append(order)

    def execute_pending(self, event: MarketEvent) -> Iterable[FillEvent]:
        if not self._pending:
            return []

        fills: list[FillEvent] = []
        for order in self._pending:
            bars = self.data.get_latest_bars(order.symbol, n=1)
            if bars.empty:
                continue
            bar = bars.iloc[0]
            base_price = float(bar.get("Open", float("nan")))
            if not (base_price == base_price) or base_price <= 0:
                continue

            slip = self.slippage_model.slippage(order, bar)
            fill_price = base_price + slip
            commission = self.commission_model.commission(order, fill_price)

            fills.append(FillEvent(
                timestamp=event.timestamp,
                symbol=order.symbol,
                quantity=order.quantity,
                direction=order.direction,
                fill_price=fill_price,
                commission=commission,
                slippage=slip,
            ))
        self._pending.clear()
        return fills