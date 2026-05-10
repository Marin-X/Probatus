"""Side-by-side: zero-cost (M1) vs realistic-cost (M2 IBKR + 2bp spread).

Same strategy, same data, same period — only the cost models differ.
The gap between the two equity curves is what M1 was hiding from you.
"""
import matplotlib.pyplot as plt

from backtester import (
    BacktestEngine,
    FixedFractionPortfolio,
    HalfSpreadSlippage,
    NextBarOpenExecutor,
    PerShareCommission,
    SMACrossover,
    YahooDataHandler,
)


def run_backtest(commission_model=None, slippage_model=None):
    """One full backtest. Returns the equity curve DataFrame."""
    symbols = ["AAPL", "MSFT", "SPY"]
    data = YahooDataHandler(symbols, start="2018-01-01", end="2024-12-31")
    strategy = SMACrossover(data, fast=20, slow=50)
    portfolio = FixedFractionPortfolio(
        data, initial_cash=100_000, allocation_fraction=0.30,
    )
    execution = NextBarOpenExecutor(
        data,
        commission_model=commission_model,
        slippage_model=slippage_model,
    )
    engine = BacktestEngine(data, strategy, portfolio, execution)
    engine.run()
    return portfolio.equity_curve(), portfolio.initial_cash


def summarize(curve, initial):
    final_eq = curve["equity"].iloc[-1]
    total_ret = final_eq / initial - 1
    n_years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (final_eq / initial) ** (1 / n_years) - 1
    peak = curve["equity"].cummax()
    max_dd = ((curve["equity"] - peak) / peak).min()
    return final_eq, total_ret, cagr, max_dd


def main():
    print("Running zero-cost backtest...")
    curve_zero, init = run_backtest()
    eq0, ret0, cagr0, dd0 = summarize(curve_zero, init)

    print("Running realistic-cost backtest (IBKR Pro + 2bp half-spread)...")
    curve_real, _ = run_backtest(
        commission_model=PerShareCommission(),
        slippage_model=HalfSpreadSlippage(half_spread_bps=2.0),
    )
    eq1, ret1, cagr1, dd1 = summarize(curve_real, init)

    print(f"\n{'='*60}")
    print(f"{'Metric':<25s}{'Zero cost':>15s}{'Realistic':>15s}")
    print(f"{'-'*60}")
    print(f"{'Final equity':<25s}${eq0:>13,.0f}${eq1:>13,.0f}")
    print(f"{'Total return':<25s}{ret0:>14.2%}{ret1:>14.2%}")
    print(f"{'CAGR':<25s}{cagr0:>14.2%}{cagr1:>14.2%}")
    print(f"{'Max drawdown':<25s}{dd0:>14.2%}{dd1:>14.2%}")
    print(f"{'='*60}")
    drag = eq0 - eq1
    print(f"Cost drag: ${drag:,.2f} over the backtest "
          f"({drag / init * 100:.2f}% of starting capital)")

    fig, ax = plt.subplots(figsize=(11, 5))
    curve_zero["equity"].plot(ax=ax, color="#3ECB7A", linewidth=1.5,
                              label=f"Zero cost  ({ret0:.1%})")
    curve_real["equity"].plot(ax=ax, color="#F5D778", linewidth=1.5,
                              label=f"IBKR + 2bp ({ret1:.1%})")
    ax.axhline(init, color="gray", linestyle="--", alpha=0.5,
               label="Initial capital")
    ax.set_title("Cost drag — SMA crossover, AAPL/MSFT/SPY 2018-2024",
                 fontsize=12)
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("compare_costs.png", dpi=120)
    print("Saved compare_costs.png")


if __name__ == "__main__":
    main()