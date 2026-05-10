"""Trading strategy implementations.

Each strategy implements a different *philosophy*:

    SMACrossover            - trend following (simple moving averages)
    BollingerMeanReversion  - mean reversion (Bollinger bands)
    DonchianBreakout        - breakout / momentum (Donchian channels)
    LeledcExhaustion        - exhaustion reversal (Pine: "LeveLeledc" by InSilico)
    EnsembleStrategy        - voting ensemble of multiple strategies

All implement the Strategy ABC and emit SignalEvents with direction
LONG / EXIT. Sizing is the Portfolio's job; strategies emit intent only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .data import DataHandler
from .events import MarketEvent, SignalEvent


class Strategy(ABC):
    def __init__(self, data: DataHandler):
        self.data = data

    @abstractmethod
    def on_market(self, event: MarketEvent) -> Iterable[SignalEvent]:
        """Called once per bar; yield zero or more SignalEvents."""


# --- Trend following: dual SMA crossover ------------------------------------

class SMACrossover(Strategy):
    """Dual-moving-average crossover (long-only).

    Goes long when fast SMA crosses above slow SMA; exits when fast
    crosses back below.
    """

    def __init__(self, data: DataHandler, fast: int = 20, slow: int = 50):
        super().__init__(data)
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self._invested: dict[str, bool] = {s: False for s in data.symbols}

    def on_market(self, event: MarketEvent) -> Iterable[SignalEvent]:
        signals: list[SignalEvent] = []
        for sym in self.data.symbols:
            bars = self.data.get_latest_bars(sym, n=self.slow + 1)
            if len(bars) < self.slow + 1:
                continue
            closes = bars["Close"]
            fast_ma = closes.tail(self.fast).mean()
            slow_ma = closes.tail(self.slow).mean()
            prev = closes.iloc[:-1]
            prev_fast = prev.tail(self.fast).mean()
            prev_slow = prev.tail(self.slow).mean()

            crossed_up = prev_fast <= prev_slow and fast_ma > slow_ma
            crossed_down = prev_fast >= prev_slow and fast_ma < slow_ma

            if crossed_up and not self._invested[sym]:
                signals.append(SignalEvent(
                    timestamp=event.timestamp, symbol=sym, direction="LONG",
                ))
                self._invested[sym] = True
            elif crossed_down and self._invested[sym]:
                signals.append(SignalEvent(
                    timestamp=event.timestamp, symbol=sym, direction="EXIT",
                ))
                self._invested[sym] = False
        return signals


# --- Mean reversion: Bollinger bands ----------------------------------------

class BollingerMeanReversion(Strategy):
    """Mean-reversion strategy using Bollinger bands (long-only).

    LONG  when close drops below the lower band (mean - num_std * sigma).
    EXIT  when close returns above the middle band (mean reverted).
    """

    def __init__(self, data: DataHandler, window: int = 20, num_std: float = 2.0):
        super().__init__(data)
        if window < 2:
            raise ValueError("window must be >= 2")
        if num_std <= 0:
            raise ValueError("num_std must be > 0")
        self.window = window
        self.num_std = float(num_std)
        self._invested: dict[str, bool] = {s: False for s in data.symbols}

    def on_market(self, event: MarketEvent) -> Iterable[SignalEvent]:
        signals: list[SignalEvent] = []
        for sym in self.data.symbols:
            bars = self.data.get_latest_bars(sym, n=self.window)
            if len(bars) < self.window:
                continue
            closes = bars["Close"].tail(self.window)
            middle = float(closes.mean())
            std = float(closes.std(ddof=0))
            if std <= 0:
                continue
            lower = middle - self.num_std * std
            current = float(closes.iloc[-1])

            if not self._invested[sym] and current < lower:
                signals.append(SignalEvent(
                    timestamp=event.timestamp, symbol=sym, direction="LONG",
                ))
                self._invested[sym] = True
            elif self._invested[sym] and current >= middle:
                signals.append(SignalEvent(
                    timestamp=event.timestamp, symbol=sym, direction="EXIT",
                ))
                self._invested[sym] = False
        return signals


# --- Breakout / momentum: Donchian channels ---------------------------------

class DonchianBreakout(Strategy):
    """Channel breakout strategy using Donchian channels (long-only).

    Foundational to Turtle Trading. Compares today's close to the
    rolling high/low of the *previous* N bars.

    LONG  when close > previous channel-bar high.
    EXIT  when close < previous exit_channel-bar low.
    """

    def __init__(self, data: DataHandler,
                 channel: int = 20, exit_channel: int = 10):
        super().__init__(data)
        if channel < 2:
            raise ValueError("channel must be >= 2")
        if exit_channel < 2:
            raise ValueError("exit_channel must be >= 2")
        self.channel = channel
        self.exit_channel = exit_channel
        self._invested: dict[str, bool] = {s: False for s in data.symbols}

    def on_market(self, event: MarketEvent) -> Iterable[SignalEvent]:
        signals: list[SignalEvent] = []
        need = max(self.channel, self.exit_channel) + 1
        for sym in self.data.symbols:
            bars = self.data.get_latest_bars(sym, n=need)
            if len(bars) < need:
                continue

            current_close = float(bars["Close"].iloc[-1])
            prev_high = float(bars["High"].iloc[-(self.channel + 1):-1].max())
            prev_low = float(bars["Low"].iloc[-(self.exit_channel + 1):-1].min())

            if not self._invested[sym] and current_close > prev_high:
                signals.append(SignalEvent(
                    timestamp=event.timestamp, symbol=sym, direction="LONG",
                ))
                self._invested[sym] = True
            elif self._invested[sym] and current_close < prev_low:
                signals.append(SignalEvent(
                    timestamp=event.timestamp, symbol=sym, direction="EXIT",
                ))
                self._invested[sym] = False
        return signals


# --- Exhaustion reversal: Leledc Exhaustion Bars ---------------------------

class LeledcExhaustion(Strategy):
    """Exhaustion-based reversal strategy (long-only).

    Python port of InSilico's "LeveLeledc" Pine Script indicator. The core
    idea: track two counters per symbol —

        bindex  — incremented every bar where close > close[4]
        sindex  — incremented every bar where close < close[4]

    A *bullish exhaustion* fires when ALL three conditions stack:

        sindex > bars                       # extended bearish run
        close > open                        # today reverses bullish
        low <= rolling low(length)          # at a new swing low

    On bullish exhaustion: enter LONG, reset sindex to 0.
    On bearish exhaustion (mirror conditions): EXIT, reset bindex to 0.

    Distinct from Bollinger mean reversion: rather than using statistical
    bands (mean ± k·sigma), Leledc uses momentum exhaustion (counters)
    plus structural confirmation (swing extreme) plus single-bar reversal
    (candle color flip). Three independent signals stacked.

    Defaults follow the original indicator: ``length=40``, ``bars=10``.
    """

    def __init__(self, data: DataHandler, length: int = 40, bars: int = 10):
        super().__init__(data)
        if length < 5:
            raise ValueError("length must be >= 5")
        if bars < 1:
            raise ValueError("bars must be >= 1")
        self.length = length
        self.bars = bars
        self._invested: dict[str, bool] = {s: False for s in data.symbols}
        self._bindex: dict[str, int] = {s: 0 for s in data.symbols}
        self._sindex: dict[str, int] = {s: 0 for s in data.symbols}

    def on_market(self, event: MarketEvent) -> Iterable[SignalEvent]:
        signals: list[SignalEvent] = []
        need = max(self.length, 5)

        for sym in self.data.symbols:
            df = self.data.get_latest_bars(sym, n=need)
            if len(df) < 5:
                continue

            close_today = float(df["Close"].iloc[-1])
            close_4ago = float(df["Close"].iloc[-5])

            if close_today > close_4ago:
                self._bindex[sym] += 1
            elif close_today < close_4ago:
                self._sindex[sym] += 1

            if len(df) < self.length:
                continue

            open_today = float(df["Open"].iloc[-1])
            high_today = float(df["High"].iloc[-1])
            low_today = float(df["Low"].iloc[-1])
            highest_high = float(df["High"].tail(self.length).max())
            lowest_low = float(df["Low"].tail(self.length).min())

            bearish_exhaustion = (
                    self._bindex[sym] > self.bars
                    and close_today < open_today
                    and high_today >= highest_high
            )
            bullish_exhaustion = (
                    self._sindex[sym] > self.bars
                    and close_today > open_today
                    and low_today <= lowest_low
            )

            # Bearish exhaustion → exit if invested
            if bearish_exhaustion:
                self._bindex[sym] = 0
                if self._invested[sym]:
                    signals.append(SignalEvent(
                        timestamp=event.timestamp, symbol=sym, direction="EXIT",
                    ))
                    self._invested[sym] = False

            # Bullish exhaustion → enter long if not already
            if bullish_exhaustion:
                self._sindex[sym] = 0
                if not self._invested[sym]:
                    signals.append(SignalEvent(
                        timestamp=event.timestamp, symbol=sym, direction="LONG",
                    ))
                    self._invested[sym] = True
        return signals


# --- Ensemble: vote-based combination of multiple strategies ---------------

class EnsembleStrategy(Strategy):
    """Vote-based ensemble of multiple strategies (long-only).

    Each bar, every sub-strategy processes the market event and
    advances its internal "invested" state via its own LONG / EXIT
    signals. The ensemble then takes a majority vote: it goes long when
    at least ``min_votes`` sub-strategies are currently long, and exits
    when fewer than ``min_votes`` are long.

    Tests whether *philosophical diversification* (combining trend
    following + mean reversion + breakout) adds alpha beyond any single
    strategy. Analogous to portfolio diversification across assets, but
    at the strategy level — strategies as the units of diversification,
    not assets.

    Parameters:
        sub_strategies  - list of Strategy instances sharing the same
                          DataHandler as this ensemble.
        min_votes       - threshold to go long (default 2). Setting to 1
                          is OR (any sub-strategy says go); setting to
                          len(subs) is AND (all must agree).
    """

    def __init__(self, data: DataHandler,
                 sub_strategies: list[Strategy], min_votes: int = 2):
        super().__init__(data)
        if not sub_strategies:
            raise ValueError("must provide at least one sub-strategy")
        if not 1 <= min_votes <= len(sub_strategies):
            raise ValueError(
                f"min_votes must be in [1, {len(sub_strategies)}]"
            )
        self.sub_strategies = list(sub_strategies)
        self.min_votes = min_votes
        # One per sub-strategy: its currently-invested state per symbol
        self._sub_invested: list[dict[str, bool]] = [
            {s: False for s in data.symbols} for _ in sub_strategies
        ]
        self._invested: dict[str, bool] = {s: False for s in data.symbols}

    def on_market(self, event: MarketEvent) -> Iterable[SignalEvent]:
        # 1. Let each sub-strategy update its internal state by processing
        #    the bar. We track their invested-state from their emitted signals.
        for i, strat in enumerate(self.sub_strategies):
            for sig in strat.on_market(event):
                if sig.direction == "LONG":
                    self._sub_invested[i][sig.symbol] = True
                elif sig.direction == "EXIT":
                    self._sub_invested[i][sig.symbol] = False

        # 2. Aggregate by majority vote per symbol.
        signals: list[SignalEvent] = []
        for sym in self.data.symbols:
            votes = sum(
                1 for sub in self._sub_invested if sub.get(sym, False)
            )
            if votes >= self.min_votes and not self._invested[sym]:
                signals.append(SignalEvent(
                    timestamp=event.timestamp, symbol=sym, direction="LONG",
                ))
                self._invested[sym] = True
            elif votes < self.min_votes and self._invested[sym]:
                signals.append(SignalEvent(
                    timestamp=event.timestamp, symbol=sym, direction="EXIT",
                ))
                self._invested[sym] = False
        return signals