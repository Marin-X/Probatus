"""Parameter sweep: run SMA crossover for a grid of (fast, slow) combinations,
plot a Sharpe heatmap.

If the strategy is robust, a broad region of the heatmap will show
positive Sharpe. If only one cell wins, the strategy is overfit.
"""
import matplotlib.pyplot as plt
import numpy as np

from backtester import (
    HalfSpreadSlippage,
    PerShareCommission,
    sma_parameter_sweep,
)


def main():
    # Manageable grid. Each backtest is ~3-5s on this machine.
    # 12 valid combos = ~1 minute total.
    fast_values = [10, 20, 30, 50]
    slow_values = [50, 100, 150, 200]

    sweep = sma_parameter_sweep(
        symbols=["AAPL", "MSFT", "SPY"],
        start="2018-01-01",
        end="2024-12-31",
        fast_values=fast_values,
        slow_values=slow_values,
        initial_cash=100_000,
        allocation_fraction=0.30,
        commission_model=PerShareCommission(),
        slippage_model=HalfSpreadSlippage(half_spread_bps=2.0),
    )

    print("\nFull sweep results (sorted by Sharpe):")
    print(sweep.sort_values("sharpe", ascending=False).to_string(index=False))

    # Pivot for heatmap: rows = fast, columns = slow, values = sharpe
    pivot_sharpe = sweep.pivot(index="fast", columns="slow", values="sharpe")
    pivot_cagr = sweep.pivot(index="fast", columns="slow", values="cagr")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, pivot, title, fmt in [
        (ax1, pivot_sharpe, "Sharpe ratio", "{:+.2f}"),
        (ax2, pivot_cagr,   "CAGR",         "{:+.1%}"),
    ]:
        # Mask invalid (fast >= slow) cells
        masked = np.ma.masked_invalid(pivot.values)
        im = ax.imshow(masked, aspect="auto", origin="lower", cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("slow window")
        ax.set_ylabel("fast window")
        ax.set_title(title)
        for i, _ in enumerate(pivot.index):
            for j, _ in enumerate(pivot.columns):
                v = pivot.iloc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=9, color="black")
        fig.colorbar(im, ax=ax)

    fig.suptitle("SMA crossover parameter sweep, AAPL/MSFT/SPY 2018-2024",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("parameter_sweep.png", dpi=120)
    print("\nSaved parameter_sweep.png")


if __name__ == "__main__":
    main()