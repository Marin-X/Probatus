"""Full end-to-end backtest with realistic costs, full analytics, and
SPY buy-and-hold benchmark comparison.
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
from backtester.analytics import (
    compare_to_benchmark,
    compute_metrics,
    compute_trade_stats,
    reconstruct_trades,
)


def main():
    symbols = ["AAPL", "MSFT", "SPY"]
    data = YahooDataHandler(symbols, start="2018-01-01", end="2024-12-31")

    strategy = SMACrossover(data, fast=20, slow=50)
    portfolio = FixedFractionPortfolio(
        data, initial_cash=100_000, allocation_fraction=0.30,
    )
    execution = NextBarOpenExecutor(
        data,
        commission_model=PerShareCommission(),
        slippage_model=HalfSpreadSlippage(half_spread_bps=2.0),
    )

    engine = BacktestEngine(data, strategy, portfolio, execution, verbose=False)
    engine.run()

    curve = portfolio.equity_curve()
    metrics = compute_metrics(curve, risk_free_rate=0.0)
    trades = reconstruct_trades(portfolio.fills)
    stats = compute_trade_stats(trades)

    # SPY buy-and-hold benchmark
    spy_prices = data.get_symbol_history("SPY")["Close"]
    bench = compare_to_benchmark(curve, spy_prices, benchmark_name="SPY",
                                 risk_free_rate=0.0)

    print(metrics)
    print()
    print(stats)
    print()
    print(bench)

    # SPY benchmark equity curve aligned to strategy curve
    spy_aligned = spy_prices.reindex(curve.index, method="ffill")
    spy_equity = portfolio.initial_cash * spy_aligned / spy_aligned.iloc[0]

    peak = curve["equity"].cummax()
    drawdown = (curve["equity"] - peak) / peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    curve["equity"].plot(ax=ax1, color="#3ECB7A", linewidth=1.5,
                         label=f"SMA(20/50) ({metrics.total_return:.1%})")
    spy_equity.plot(ax=ax1, color="#888888", linewidth=1.2, linestyle="--",
                    label=f"SPY buy & hold ({bench.benchmark_total_return:.1%})")
    ax1.axhline(portfolio.initial_cash, color="gray", linestyle=":",
                alpha=0.5)
    ax1.set_title(
        f"SMA crossover vs SPY — "
        f"alpha {bench.alpha_annual:+.2%}/yr, "
        f"beta {bench.beta:.2f}, "
        f"IR {bench.information_ratio:.2f}",
        fontsize=12)
    ax1.set_ylabel("Equity ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    drawdown.plot(ax=ax2, color="#F5D778", linewidth=1.0)
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#F5D778", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.tight_layout()
    fig.savefig("equity_curve.png", dpi=120)
    print("\nSaved equity_curve.png")


if __name__ == "__main__":
    main()