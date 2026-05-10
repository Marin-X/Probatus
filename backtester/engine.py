"""The backtest engine.

Orchestrates the four components in the correct order. The loop is
intentionally simple — every interesting feature (costs, attribution,
walk-forward, etc.) lives in the components, not here.

Per-bar order of operations:

    1. Pull next MarketEvent from data handler. Stop if None.
    2. Fill any pending orders at THIS bar's OPEN.
       This is what enforces no look-ahead leakage in execution.
    3. Mark portfolio to market with this bar's CLOSE (snapshot).
    4. Strategy reacts to the bar; emits SignalEvents.
    5. Drain the queue: signals -> sized orders -> queued for next bar.
"""
from __future__ import annotations

from queue import Queue

from .data import DataHandler
from .events import OrderEvent, SignalEvent
from .execution import ExecutionHandler
from .portfolio import Portfolio
from .strategy import Strategy


class BacktestEngine:
    def __init__(self,
                 data: DataHandler,
                 strategy: Strategy,
                 portfolio: Portfolio,
                 execution: ExecutionHandler,
                 verbose: bool = False):
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.verbose = verbose
        self.queue: "Queue" = Queue()
        self.bars_processed: int = 0

    def run(self) -> None:
        while True:
            market_event = self.data.update()
            if market_event is None:
                break
            self.bars_processed += 1

            # 1. Fill any pending orders at this bar's OPEN.
            for fill in self.execution.execute_pending(market_event):
                self.portfolio.on_fill(fill)
                if self.verbose:
                    print(f"[FILL] {fill.timestamp.date()} {fill.direction} "
                          f"{fill.quantity} {fill.symbol} @ {fill.fill_price:.2f}")

            # 2. Mark portfolio to market with this bar's CLOSE.
            self.portfolio.update_timeindex(market_event)

            # 3. Strategy reacts to the new bar.
            for signal in self.strategy.on_market(market_event):
                self.queue.put(signal)

            # 4. Drain the queue: signals -> orders queued for next bar.
            while not self.queue.empty():
                event = self.queue.get()
                if isinstance(event, SignalEvent):
                    for order in self.portfolio.on_signal(event):
                        self.queue.put(order)
                elif isinstance(event, OrderEvent):
                    self.execution.queue_order(event)

        if self.verbose:
            print(f"\nBacktest complete. {self.bars_processed} bars processed.")