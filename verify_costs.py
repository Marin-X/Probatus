"""Sanity check cost models in isolation: construct an order + bar,
print what each model charges.
"""
from datetime import datetime

import pandas as pd

from backtester.costs import (
    HalfSpreadSlippage,
    NoSlippage,
    PercentCommission,
    PerShareCommission,
    ZeroCommission,
)
from backtester.events import OrderEvent


def main():
    order = OrderEvent(
        timestamp=datetime(2024, 6, 15),
        symbol="AAPL",
        order_type="MKT",
        quantity=100,
        direction="BUY",
    )
    bar = pd.Series({"Open": 200.0, "High": 202.0, "Low": 199.0,
                     "Close": 201.0, "Volume": 50_000_000})
    notional = order.quantity * bar["Open"]

    print(f"Order: BUY {order.quantity} {order.symbol} @ open=${bar['Open']}")
    print(f"Notional: ${notional:,.2f}\n")

    print(f"{'Commission models':<40s}{'$':>10s}{'bps':>8s}")
    print("-" * 58)
    for name, model in [
        ("ZeroCommission",                        ZeroCommission()),
        ("PerShareCommission (IBKR Pro)",         PerShareCommission()),
        ("PercentCommission (Coinbase, 0.1%)",    PercentCommission(0.001)),
    ]:
        c = model.commission(order, fill_price=bar["Open"])
        print(f"{name:<40s}{c:>10.4f}{c / notional * 1e4:>8.2f}")

    print(f"\n{'Slippage models':<40s}{'$/sh':>10s}{'$total':>10s}")
    print("-" * 60)
    for name, model in [
        ("NoSlippage",                  NoSlippage()),
        ("HalfSpreadSlippage(1 bp)",    HalfSpreadSlippage(1.0)),
        ("HalfSpreadSlippage(5 bp)",    HalfSpreadSlippage(5.0)),
    ]:
        s = model.slippage(order, bar)
        total = s * order.quantity
        print(f"{name:<40s}{s:>+10.4f}{total:>+10.4f}")


if __name__ == "__main__":
    main()