"""Sanity check: feed the portfolio one LONG signal and one fill manually,
watch the cash and positions update.
"""
from backtester.data import YahooDataHandler
from backtester.events import FillEvent, SignalEvent
from backtester.portfolio import FixedFractionPortfolio


def main():
    data = YahooDataHandler(["AAPL"], start="2024-01-01", end="2024-02-15")
    portfolio = FixedFractionPortfolio(
        data, initial_cash=100_000, allocation_fraction=0.30,
    )

    # Advance the data handler 5 bars (so we have prices to work with).
    for _ in range(5):
        e = data.update()
        portfolio.update_timeindex(e)

    print(f"Initial state:")
    print(f"  Cash      = ${portfolio.cash:,.2f}")
    print(f"  Positions = {dict(portfolio.positions)}")
    print(f"  Equity    = ${portfolio.history[-1].total_equity:,.2f}")
    print(f"  AAPL @     ${data.get_latest_price('AAPL'):.2f}")

    # Strategy emits a LONG signal -> portfolio sizes it.
    signal = SignalEvent(timestamp=data.current_time, symbol="AAPL",
                         direction="LONG")
    orders = list(portfolio.on_signal(signal))
    print(f"\nLONG signal -> {len(orders)} order(s):")
    for o in orders:
        print(f"  {o.direction} {o.quantity} {o.symbol} @ {o.order_type}")

    # Engine advances one bar; executor fills at next bar's open.
    next_event = data.update()
    fill_price = data.get_latest_price("AAPL", field="Open")
    fill = FillEvent(
        timestamp=next_event.timestamp,
        symbol=orders[0].symbol,
        quantity=orders[0].quantity,
        direction=orders[0].direction,
        fill_price=fill_price,
    )
    portfolio.on_fill(fill)
    portfolio.update_timeindex(next_event)

    print(f"\nAfter fill at ${fill_price:.2f}:")
    print(f"  Cash      = ${portfolio.cash:,.2f}")
    print(f"  Positions = {dict(portfolio.positions)}")
    print(f"  Equity    = ${portfolio.history[-1].total_equity:,.2f}")

    # Mark to market for a few more bars.
    for _ in range(10):
        e = data.update()
        if e is None:
            break
        portfolio.update_timeindex(e)

    print(f"\nEquity curve (last 5 rows):")
    print(portfolio.equity_curve().round(2).tail(5))


if __name__ == "__main__":
    main()