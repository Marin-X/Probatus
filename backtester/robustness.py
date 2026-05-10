"""Robustness tests: parameter sweeps, walk-forward analysis.

The core ``parameter_sweep`` and ``walk_forward`` functions are
strategy-agnostic — they take a ``strategy_factory`` callable plus a
``param_grid`` dict and work for any strategy. Strategy-specific
wrappers (e.g. ``sma_parameter_sweep``) build the factory and forward.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from .analytics import compute_metrics
from .costs import CommissionModel, SlippageModel
from .data import InMemoryDataHandler, YahooDataHandler
from .engine import BacktestEngine
from .events import FillEvent
from .execution import NextBarOpenExecutor
from .portfolio import FixedFractionPortfolio
from .strategy import (
    BollingerMeanReversion,
    DonchianBreakout,
    SMACrossover,
    Strategy,
)


# Type aliases
StrategyFactory = Callable[..., Strategy]                  # (data, **params) -> Strategy
ValidityFn = Callable[..., bool]                           # (**params) -> bool
ProgressCallback = Callable[[int, int, str], None]         # (current, total, msg)


# --- Internal helpers -------------------------------------------------------

def _backtest_window(
        raw_data: dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
        strategy_factory: StrategyFactory,
        strategy_params: dict[str, Any],
        initial_cash: float,
        allocation_fraction: float,
        commission_model: Optional[CommissionModel],
        slippage_model: Optional[SlippageModel],
) -> tuple[pd.DataFrame, list[FillEvent]]:
    sliced = {
        sym: df[(df.index >= start) & (df.index <= end)]
        for sym, df in raw_data.items()
    }
    if any(df.empty for df in sliced.values()):
        return pd.DataFrame(), []

    data = InMemoryDataHandler(sliced)
    try:
        strategy = strategy_factory(data, **strategy_params)
    except Exception:
        return pd.DataFrame(), []
    portfolio = FixedFractionPortfolio(
        data, initial_cash=initial_cash,
        allocation_fraction=allocation_fraction,
    )
    execution = NextBarOpenExecutor(
        data,
        commission_model=commission_model,
        slippage_model=slippage_model,
    )
    engine = BacktestEngine(data, strategy, portfolio, execution, verbose=False)
    engine.run()
    return portfolio.equity_curve(), portfolio.fills


def _expand_grid(
        param_grid: dict[str, Iterable[Any]],
        is_valid_combination: Optional[ValidityFn] = None,
) -> list[dict[str, Any]]:
    """Cartesian product of param_grid, filtered through is_valid_combination."""
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    out = []
    for combo in itertools.product(*value_lists):
        params = dict(zip(keys, combo))
        if is_valid_combination is None or is_valid_combination(**params):
            out.append(params)
    return out


# --- Generic parameter sweep ------------------------------------------------

def parameter_sweep(
        symbols: list[str],
        start: str,
        end: str,
        strategy_factory: StrategyFactory,
        param_grid: dict[str, Iterable[Any]],
        is_valid_combination: Optional[ValidityFn] = None,
        initial_cash: float = 100_000,
        allocation_fraction: float = 0.30,
        commission_model: Optional[CommissionModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        verbose: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Run a backtest for every valid parameter combination in the grid."""
    if verbose:
        print(f"Downloading data for {symbols} ({start} -> {end})...")
    if progress_callback:
        progress_callback(0, 1, "downloading data...")
    seed = YahooDataHandler(symbols, start=start, end=end)
    raw_data = {sym: seed.get_symbol_history(sym) for sym in symbols}

    combos = _expand_grid(param_grid, is_valid_combination)
    total = len(combos)
    if verbose:
        print(f"Running {total} backtests...\n")

    rows: list[dict] = []
    for i, params in enumerate(combos, start=1):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        if progress_callback:
            progress_callback(i, total, param_str)
        curve, fills = _backtest_window(
            raw_data, pd.Timestamp(start), pd.Timestamp(end),
            strategy_factory, params,
            initial_cash, allocation_fraction,
            commission_model, slippage_model,
        )
        if curve.empty or len(curve) < 2:
            continue
        m = compute_metrics(curve)
        rows.append({
            **params,
            "total_return": m.total_return, "cagr": m.cagr,
            "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
            "max_dd": m.max_drawdown, "calmar": m.calmar_ratio,
            "vol": m.annual_volatility,
            "n_trades": len(fills) // 2,
        })
        if verbose:
            print(f"  [{i:>2d}/{total}] {param_str}  "
                  f"Sharpe={m.sharpe_ratio:+.2f}  CAGR={m.cagr:+.1%}  "
                  f"DD={m.max_drawdown:+.1%}")
    return pd.DataFrame(rows)


# --- Strategy-specific wrappers --------------------------------------------

def sma_parameter_sweep(
        symbols: list[str], start: str, end: str,
        fast_values: Iterable[int], slow_values: Iterable[int],
        **kwargs,
) -> pd.DataFrame:
    return parameter_sweep(
        symbols, start, end,
        strategy_factory=lambda data, fast, slow: SMACrossover(data, fast=fast, slow=slow),
        param_grid={"fast": list(fast_values), "slow": list(slow_values)},
        is_valid_combination=lambda fast, slow: fast < slow,
        **kwargs,
    )


def bollinger_parameter_sweep(
        symbols: list[str], start: str, end: str,
        window_values: Iterable[int], num_std_values: Iterable[float],
        **kwargs,
) -> pd.DataFrame:
    return parameter_sweep(
        symbols, start, end,
        strategy_factory=lambda data, window, num_std: BollingerMeanReversion(
            data, window=window, num_std=num_std,
        ),
        param_grid={"window": list(window_values),
                    "num_std": list(num_std_values)},
        **kwargs,
    )


def donchian_parameter_sweep(
        symbols: list[str], start: str, end: str,
        channel_values: Iterable[int], exit_channel_values: Iterable[int],
        **kwargs,
) -> pd.DataFrame:
    return parameter_sweep(
        symbols, start, end,
        strategy_factory=lambda data, channel, exit_channel: DonchianBreakout(
            data, channel=channel, exit_channel=exit_channel,
        ),
        param_grid={"channel": list(channel_values),
                    "exit_channel": list(exit_channel_values)},
        is_valid_combination=lambda channel, exit_channel: exit_channel <= channel,
        **kwargs,
    )


# --- Walk-forward analysis -------------------------------------------------

@dataclass
class WalkForwardFold:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    test_cagr: float
    test_total_return: float
    test_max_dd: float
    test_n_trades: int
    test_curve: pd.DataFrame


def walk_forward(
        symbols: list[str],
        start: str,
        end: str,
        strategy_factory: StrategyFactory,
        param_grid: dict[str, Iterable[Any]],
        is_valid_combination: Optional[ValidityFn] = None,
        train_years: float = 3.0,
        test_years: float = 1.0,
        initial_cash: float = 100_000,
        allocation_fraction: float = 0.30,
        commission_model: Optional[CommissionModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        verbose: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
) -> list[WalkForwardFold]:
    """Walk-forward optimization for any strategy."""
    if verbose:
        print(f"Downloading data for {symbols} ({start} -> {end})...")
    if progress_callback:
        progress_callback(0, 1, "downloading data...")
    seed = YahooDataHandler(symbols, start=start, end=end)
    raw_data = {sym: seed.get_symbol_history(sym) for sym in symbols}

    overall_start = pd.Timestamp(start)
    overall_end = pd.Timestamp(end)
    combos = _expand_grid(param_grid, is_valid_combination)

    # Build fold windows
    windows = []
    train_start = overall_start
    fold_idx = 0
    while True:
        train_end = train_start + pd.Timedelta(days=int(train_years * 365.25))
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=int(test_years * 365.25))
        if test_end > overall_end:
            break
        fold_idx += 1
        windows.append((fold_idx, train_start, train_end, test_start, test_end))
        train_start = train_start + pd.Timedelta(days=int(test_years * 365.25))

    total_steps = len(windows) * len(combos) + len(windows)
    step = 0

    folds: list[WalkForwardFold] = []
    for fold_idx, t_start, t_end, ts_start, ts_end in windows:
        if verbose:
            print(f"Fold {fold_idx}: train {t_start.date()}->{t_end.date()}  |  "
                  f"test {ts_start.date()}->{ts_end.date()}")

        best: Optional[tuple[float, dict[str, Any]]] = None
        for params in combos:
            step += 1
            if progress_callback:
                pstr = ", ".join(f"{k}={v}" for k, v in params.items())
                progress_callback(step, total_steps,
                                  f"fold {fold_idx} train [{pstr}]")
            curve, _ = _backtest_window(
                raw_data, t_start, t_end,
                strategy_factory, params,
                initial_cash, allocation_fraction,
                commission_model, slippage_model,
            )
            if curve.empty or len(curve) < 2:
                continue
            m = compute_metrics(curve)
            if best is None or m.sharpe_ratio > best[0]:
                best = (m.sharpe_ratio, params)

        if best is None:
            step += 1
            if verbose:
                print(f"  No valid params; skipping fold")
            continue

        train_sharpe, best_params = best
        step += 1
        if progress_callback:
            pstr = ", ".join(f"{k}={v}" for k, v in best_params.items())
            progress_callback(step, total_steps,
                              f"fold {fold_idx} test [{pstr}]")

        full_curve, full_fills = _backtest_window(
            raw_data, overall_start, ts_end,
            strategy_factory, best_params,
            initial_cash, allocation_fraction,
            commission_model, slippage_model,
        )
        test_curve = full_curve[
            (full_curve.index >= ts_start) & (full_curve.index <= ts_end)
            ].copy()
        if test_curve.empty or len(test_curve) < 2:
            if verbose:
                print(f"  Empty test window; skipping")
            continue

        scale = initial_cash / float(test_curve["equity"].iloc[0])
        for col in ("equity", "cash", "positions_value"):
            if col in test_curve.columns:
                test_curve[col] = test_curve[col] * scale

        test_metrics = compute_metrics(test_curve)
        n_trades_in_test = sum(
            1 for f in full_fills
            if f.timestamp >= ts_start and f.direction == "BUY"
        )

        folds.append(WalkForwardFold(
            fold_idx=fold_idx,
            train_start=t_start, train_end=t_end,
            test_start=ts_start, test_end=ts_end,
            best_params=best_params,
            train_sharpe=float(train_sharpe),
            test_sharpe=test_metrics.sharpe_ratio,
            test_cagr=test_metrics.cagr,
            test_total_return=test_metrics.total_return,
            test_max_dd=test_metrics.max_drawdown,
            test_n_trades=n_trades_in_test,
            test_curve=test_curve,
        ))

        if verbose:
            pstr = ", ".join(f"{k}={v}" for k, v in best_params.items())
            print(f"  Best: [{pstr}]  IS Sharpe {train_sharpe:+.2f}  ->  "
                  f"OOS Sharpe {test_metrics.sharpe_ratio:+.2f}  "
                  f"OOS CAGR {test_metrics.cagr:+.1%}")

    return folds


# Backward-compat: walk_forward_analysis is the SMA wrapper

def walk_forward_analysis(
        symbols: list[str], start: str, end: str,
        fast_values: Iterable[int], slow_values: Iterable[int],
        **kwargs,
) -> list[WalkForwardFold]:
    return walk_forward(
        symbols, start, end,
        strategy_factory=lambda data, fast, slow: SMACrossover(data, fast=fast, slow=slow),
        param_grid={"fast": list(fast_values), "slow": list(slow_values)},
        is_valid_combination=lambda fast, slow: fast < slow,
        **kwargs,
    )