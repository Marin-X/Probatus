"""Performance analytics computed from a portfolio's equity curve and fills."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .events import FillEvent


TRADING_DAYS_PER_YEAR = 252


# --- Equity-curve metrics ----------------------------------------------------

@dataclass
class PerformanceMetrics:
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    n_days: int
    final_equity: float
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    best_day: float
    worst_day: float
    pct_positive_days: float

    def __str__(self) -> str:
        return (
            f"┌─ Performance ─────────────────────────────────────┐\n"
            f"│  Period       {self.period_start.date()} -> {self.period_end.date()}  ({self.n_days} days)\n"
            f"│  Final equity ${self.final_equity:>15,.2f}\n"
            f"│  Total return {self.total_return:>16.2%}\n"
            f"│  CAGR         {self.cagr:>16.2%}\n"
            f"├─ Risk-adjusted ──────────────────────────────────┤\n"
            f"│  Annual vol   {self.annual_volatility:>16.2%}\n"
            f"│  Sharpe       {self.sharpe_ratio:>16.3f}\n"
            f"│  Sortino      {self.sortino_ratio:>16.3f}\n"
            f"│  Calmar       {self.calmar_ratio:>16.3f}\n"
            f"├─ Drawdown ───────────────────────────────────────┤\n"
            f"│  Max DD       {self.max_drawdown:>16.2%}\n"
            f"│  Max DD days  {self.max_drawdown_duration_days:>16d}\n"
            f"├─ Daily returns ──────────────────────────────────┤\n"
            f"│  Best day     {self.best_day:>16.2%}\n"
            f"│  Worst day    {self.worst_day:>16.2%}\n"
            f"│  % positive   {self.pct_positive_days:>16.2%}\n"
            f"└──────────────────────────────────────────────────┘"
        )


def compute_metrics(curve: pd.DataFrame,
                    risk_free_rate: float = 0.0) -> PerformanceMetrics:
    if "equity" not in curve.columns:
        raise ValueError("curve must have an 'equity' column")
    equity = curve["equity"]
    if len(equity) < 2:
        raise ValueError("need at least 2 equity points to compute metrics")

    returns = equity.pct_change().dropna()
    period_start, period_end = curve.index[0], curve.index[-1]
    n_days = (period_end - period_start).days
    n_years = n_days / 365.25 if n_days > 0 else 1.0

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    annual_vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if excess.std() > 0 else 0.0
    )
    downside = returns[returns < 0]
    downside_vol = (
        float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(downside) > 1 else 0.0
    )
    sortino = (
        float(excess.mean() * TRADING_DAYS_PER_YEAR / downside_vol)
        if downside_vol > 0 else 0.0
    )

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())
    max_dd_duration = _longest_drawdown_in_days(drawdown)
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return PerformanceMetrics(
        period_start=period_start,
        period_end=period_end,
        n_days=n_days,
        final_equity=float(equity.iloc[-1]),
        total_return=total_return,
        cagr=float(cagr),
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        best_day=float(returns.max()),
        worst_day=float(returns.min()),
        pct_positive_days=float((returns > 0).mean()),
    )


def _longest_drawdown_in_days(drawdown: pd.Series) -> int:
    longest, run_start = 0, None
    for ts, dd in drawdown.items():
        if dd < 0:
            if run_start is None:
                run_start = ts
            longest = max(longest, (ts - run_start).days)
        else:
            run_start = None
    return int(longest)


# --- Trade reconstruction ----------------------------------------------------

@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    quantity: int
    direction: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commissions: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def hold_days(self) -> int:
        if self.exit_time is None:
            return 0
        return (self.exit_time - self.entry_time).days


def reconstruct_trades(fills: list[FillEvent]) -> list[Trade]:
    trades: list[Trade] = []
    open_per_symbol: dict[str, Trade] = {}

    for fill in fills:
        sym = fill.symbol
        if fill.direction == "BUY":
            open_per_symbol[sym] = Trade(
                symbol=sym,
                entry_time=fill.timestamp,
                entry_price=fill.fill_price,
                exit_time=None,
                exit_price=None,
                quantity=fill.quantity,
                direction="LONG",
                commissions=fill.commission,
            )
        elif fill.direction == "SELL":
            if sym not in open_per_symbol:
                continue
            t = open_per_symbol.pop(sym)
            t.exit_time = fill.timestamp
            t.exit_price = fill.fill_price
            t.commissions += fill.commission
            t.pnl = (t.exit_price - t.entry_price) * t.quantity - t.commissions
            t.pnl_pct = (t.exit_price / t.entry_price - 1) if t.entry_price > 0 else 0.0
            trades.append(t)

    trades.extend(open_per_symbol.values())
    return trades


@dataclass
class TradeStats:
    n_trades: int
    n_closed: int
    n_open: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_hold_days: float
    best_trade_pnl: float
    worst_trade_pnl: float
    total_pnl: float
    total_commissions: float

    def __str__(self) -> str:
        pf_str = (f"{self.profit_factor:>16.3f}" if self.profit_factor != float("inf")
                  else f"{'inf':>16s}")
        return (
            f"┌─ Trade statistics ───────────────────────────────┐\n"
            f"│  Trades       {self.n_trades:>16d}\n"
            f"│  Closed       {self.n_closed:>16d}\n"
            f"│  Still open   {self.n_open:>16d}\n"
            f"├─ Outcomes ───────────────────────────────────────┤\n"
            f"│  Wins         {self.n_wins:>16d}\n"
            f"│  Losses       {self.n_losses:>16d}\n"
            f"│  Win rate     {self.win_rate:>16.2%}\n"
            f"│  Profit factor{pf_str}\n"
            f"├─ P&L per trade ──────────────────────────────────┤\n"
            f"│  Avg win      ${self.avg_win:>15,.2f}\n"
            f"│  Avg loss     ${self.avg_loss:>15,.2f}\n"
            f"│  Best         ${self.best_trade_pnl:>15,.2f}\n"
            f"│  Worst        ${self.worst_trade_pnl:>15,.2f}\n"
            f"├─ Other ──────────────────────────────────────────┤\n"
            f"│  Avg hold     {self.avg_hold_days:>15.1f} days\n"
            f"│  Total P&L    ${self.total_pnl:>15,.2f}\n"
            f"│  Commissions  ${self.total_commissions:>15,.2f}\n"
            f"└──────────────────────────────────────────────────┘"
        )


def compute_trade_stats(trades: list[Trade]) -> TradeStats:
    closed = [t for t in trades if not t.is_open]
    n_open = len(trades) - len(closed)

    if not closed:
        return TradeStats(
            n_trades=len(trades), n_closed=0, n_open=n_open,
            n_wins=0, n_losses=0, win_rate=0.0,
            avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            avg_hold_days=0.0, best_trade_pnl=0.0, worst_trade_pnl=0.0,
            total_pnl=0.0, total_commissions=0.0,
        )

    wins = [t.pnl for t in closed if t.pnl > 0]
    losses = [t.pnl for t in closed if t.pnl <= 0]
    sum_wins = sum(wins)
    sum_losses = sum(losses)
    profit_factor = (
        abs(sum_wins / sum_losses) if sum_losses != 0 else float("inf")
    )

    return TradeStats(
        n_trades=len(trades),
        n_closed=len(closed),
        n_open=n_open,
        n_wins=len(wins),
        n_losses=len(losses),
        win_rate=len(wins) / len(closed),
        avg_win=sum_wins / len(wins) if wins else 0.0,
        avg_loss=sum_losses / len(losses) if losses else 0.0,
        profit_factor=profit_factor,
        avg_hold_days=sum(t.hold_days for t in closed) / len(closed),
        best_trade_pnl=max(t.pnl for t in closed),
        worst_trade_pnl=min(t.pnl for t in closed),
        total_pnl=sum(t.pnl for t in closed),
        total_commissions=sum(t.commissions for t in closed),
    )


# --- Benchmark comparison ----------------------------------------------------

@dataclass
class BenchmarkComparison:
    benchmark_name: str
    strategy_total_return: float
    benchmark_total_return: float
    strategy_cagr: float
    benchmark_cagr: float
    strategy_sharpe: float
    benchmark_sharpe: float
    alpha_annual: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float

    def __str__(self) -> str:
        diff = self.strategy_total_return - self.benchmark_total_return
        return (
            f"┌─ vs {self.benchmark_name + ' (buy & hold)':<46s}┐\n"
            f"│                       Strategy     Benchmark\n"
            f"│  Total return   {self.strategy_total_return:>13.2%}{self.benchmark_total_return:>14.2%}\n"
            f"│  CAGR           {self.strategy_cagr:>13.2%}{self.benchmark_cagr:>14.2%}\n"
            f"│  Sharpe         {self.strategy_sharpe:>13.3f}{self.benchmark_sharpe:>14.3f}\n"
            f"│  Difference     {diff:>+13.2%}\n"
            f"├─ Regression vs benchmark ────────────────────────┤\n"
            f"│  Beta              {self.beta:>+13.3f}\n"
            f"│  Alpha (ann.)      {self.alpha_annual:>+13.2%}\n"
            f"│  Correlation       {self.correlation:>+13.3f}\n"
            f"│  Tracking error    {self.tracking_error:>13.2%}\n"
            f"│  Information ratio {self.information_ratio:>+13.3f}\n"
            f"└──────────────────────────────────────────────────┘"
        )


def compare_to_benchmark(
        strategy_curve: pd.DataFrame,
        benchmark_prices: pd.Series,
        benchmark_name: str = "Benchmark",
        risk_free_rate: float = 0.0,
) -> BenchmarkComparison:
    """Compute strategy-vs-benchmark stats: alpha, beta, IR, etc.

    benchmark_prices is a Series of close prices indexed by date.
    """
    strat_eq = strategy_curve["equity"]
    bench = benchmark_prices.reindex(strat_eq.index, method="ffill").dropna()
    strat_eq = strat_eq.reindex(bench.index)

    strat_ret = strat_eq.pct_change().dropna()
    bench_ret = bench.pct_change().dropna()
    common = strat_ret.index.intersection(bench_ret.index)
    strat_ret = strat_ret.loc[common]
    bench_ret = bench_ret.loc[common]

    strat_total = float(strat_eq.iloc[-1] / strat_eq.iloc[0] - 1)
    bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1)
    n_years = (strat_eq.index[-1] - strat_eq.index[0]).days / 365.25
    strat_cagr = (1 + strat_total) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    bench_cagr = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    strat_sharpe = (
        float((strat_ret - daily_rf).mean() / strat_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if strat_ret.std() > 0 else 0.0
    )
    bench_sharpe = (
        float((bench_ret - daily_rf).mean() / bench_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if bench_ret.std() > 0 else 0.0
    )

    bench_var = float(bench_ret.var())
    beta = float(strat_ret.cov(bench_ret) / bench_var) if bench_var > 0 else 0.0
    daily_alpha = (strat_ret.mean() - daily_rf) - beta * (bench_ret.mean() - daily_rf)
    alpha_annual = float(daily_alpha * TRADING_DAYS_PER_YEAR)
    correlation = float(strat_ret.corr(bench_ret))

    active = strat_ret - bench_ret
    tracking_error = float(active.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    information_ratio = (
        float(active.mean() * TRADING_DAYS_PER_YEAR / tracking_error)
        if tracking_error > 0 else 0.0
    )

    return BenchmarkComparison(
        benchmark_name=benchmark_name,
        strategy_total_return=strat_total,
        benchmark_total_return=bench_total,
        strategy_cagr=float(strat_cagr),
        benchmark_cagr=float(bench_cagr),
        strategy_sharpe=strat_sharpe,
        benchmark_sharpe=bench_sharpe,
        alpha_annual=alpha_annual,
        beta=beta,
        correlation=correlation,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
    )