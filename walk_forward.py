"""Walk-forward analysis: optimize on a rolling 3-year train window,
test on the next 1-year window. Aggregate test-window results.

Tells you whether SMA parameters chosen on past data actually predict
performance on future data, or whether each "best" is just lucky.
"""
import matplotlib.pyplot as plt
import numpy as np

from backtester import (
    HalfSpreadSlippage,
    PerShareCommission,
    walk_forward_analysis,
)


def main():
    folds = walk_forward_analysis(
        symbols=["AAPL", "MSFT", "SPY"],
        start="2018-01-01",
        end="2024-12-31",
        fast_values=[10, 20, 30, 50],
        slow_values=[50, 100, 150, 200],
        train_years=3.0,
        test_years=1.0,
        initial_cash=100_000,
        allocation_fraction=0.30,
        commission_model=PerShareCommission(),
        slippage_model=HalfSpreadSlippage(half_spread_bps=2.0),
    )

    if not folds:
        print("No folds produced — check date range vs train/test windows.")
        return

    print("\n" + "=" * 110)
    print("Walk-forward summary")
    print("=" * 110)
    print(f"{'Fold':>4s}  {'Train':>22s}  {'Test':>22s}  "
          f"{'Best':>8s}  {'IS Sh':>6s}  {'OOS Sh':>7s}  "
          f"{'OOS CAGR':>9s}  {'OOS DD':>8s}  {'Trd':>4s}")
    print("-" * 110)
    for f in folds:
        print(f"{f.fold_idx:>4d}  "
              f"{f.train_start.date()}->{f.train_end.date()}  "
              f"{f.test_start.date()}->{f.test_end.date()}  "
              f"({f.best_fast:>2d},{f.best_slow:>3d}) "
              f"{f.train_sharpe:>+6.2f}  {f.test_sharpe:>+7.2f}  "
              f"{f.test_cagr:>+9.2%}  {f.test_max_dd:>+8.2%}  "
              f"{f.test_n_trades:>4d}")

    avg_is = np.mean([f.train_sharpe for f in folds])
    avg_oos = np.mean([f.test_sharpe for f in folds])
    avg_cagr = np.mean([f.test_cagr for f in folds])
    print("-" * 110)
    print(f"{'AVG':>4s}{'':>50s}"
          f"{avg_is:>+18.2f}  {avg_oos:>+7.2f}  {avg_cagr:>+9.2%}")
    print(f"\nIS -> OOS Sharpe degradation: {avg_oos - avg_is:+.2f}")
    print(f"  Within ~0.2-0.3: strategy is robust, parameters generalize")
    print(f"  OOS << IS:        parameter selection is overfitting noise")

    # Plot OOS equity curves stitched together
    fig, ax = plt.subplots(figsize=(11, 5))
    cum_equity = 100_000.0
    for f in folds:
        eq = f.test_curve["equity"]
        scale = cum_equity / eq.iloc[0]
        eq_scaled = eq * scale
        eq_scaled.plot(ax=ax, label=f"Fold {f.fold_idx}: ({f.best_fast},{f.best_slow})",
                       linewidth=1.4)
        cum_equity = float(eq_scaled.iloc[-1])

    ax.axhline(100_000, color="gray", linestyle="--", alpha=0.5,
               label="Initial capital")
    ax.set_title(f"Walk-forward out-of-sample equity — "
                 f"avg OOS Sharpe {avg_oos:.2f}, "
                 f"final equity ${cum_equity:,.0f}",
                 fontsize=12)
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("walk_forward.png", dpi=120)
    print(f"\nSaved walk_forward.png")


if __name__ == "__main__":
    main()