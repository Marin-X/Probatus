"""Show how Almgren-Chriss impact scales from retail to institutional.

Same stock, same price, same daily vol — only order size changes.
You'll see retail-sized orders barely register while institutional
trades pay meaningful impact.
"""
from datetime import datetime

import pandas as pd

from backtester.costs import AlmgrenChrissImpact
from backtester.events import OrderEvent


def main():
    bar = pd.Series({
        "Open": 200.0, "High": 202.0, "Low": 199.0, "Close": 201.0,
        "Volume": 50_000_000,  # AAPL-ish daily volume
    })
    model = AlmgrenChrissImpact(eta=0.1, sigma=0.02)

    print(f"Stock: $200, daily vol 2%, ADV 50M shares")
    print(f"Almgren-Chriss with eta=0.1\n")
    print(f"{'Order size':>12s}{'Particip.':>12s}{'Impact $/sh':>14s}{'Impact bps':>14s}{'Total cost':>14s}")
    print("-" * 70)

    for qty in [100, 1_000, 10_000, 100_000, 500_000, 1_000_000]:
        order = OrderEvent(
            timestamp=datetime(2024, 6, 15),
            symbol="AAPL",
            order_type="MKT",
            quantity=qty,
            direction="BUY",
        )
        impact = model.slippage(order, bar)
        participation = qty / bar["Volume"]
        bps = impact / bar["Open"] * 1e4
        total = impact * qty
        print(f"{qty:>12,d}{participation:>12.4%}{impact:>14.4f}{bps:>14.3f}{total:>14,.2f}")


if __name__ == "__main__":
    main()