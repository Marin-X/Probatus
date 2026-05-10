"""Sanity check: queue an order at bar T, watch it fill at T+1's OPEN.

Demonstrates the look-ahead guard — the strategy's decision price (T's
close) and the actual fill price (T+1's open) are different, and the
gap is the realistic execution cost a vectorized backtester would hide.
"""
from backtester.data import YahooDataHandler
from backtester.events import OrderEvent
from backtester.execution import NextBarOpenExecutor


def main():
    data = YahooDataHandler(["AAPL"], start="2024-01-01", end="2024-01-31")
    executor = NextBarOpenExecutor(data)

    # Advance to bar 5 (some warm-up).
    for _ in range(5):
        data.update()

    decision_time = data.current_time
    decision_price = data.get_latest_price("AAPL", field="Close")
    print(f"Bar T (decision): {decision_time.date()}")
    print(f"  Close: ${decision_price:.2f}  <- strategy decides here")

    # Strategy/portfolio generates an order at T's close.
    order = OrderEvent(
        timestamp=decision_time,
        symbol="AAPL",
        order_type="MKT",
        quantity=100,
        direction="BUY",
    )
    executor.queue_order(order)
    print(f"  Queued: BUY 100 AAPL")

    # Engine advances to T+1; executor fills.
    next_event = data.update()
    next_open = data.get_latest_price("AAPL", field="Open")
    print(f"\nBar T+1 (fill): {next_event.timestamp.date()}")
    print(f"  Open: ${next_open:.2f}  <- executor fills here")

    fills = list(executor.execute_pending(next_event))
    print(f"\n{len(fills)} fill(s):")
    for f in fills:
        gap = f.fill_price - decision_price
        print(f"  {f.direction} {f.quantity} {f.symbol} @ ${f.fill_price:.2f}  "
              f"(gap from decision: ${gap:+.2f}/share, ${gap * f.quantity:+,.2f} total)")
    print(f"\nThis gap is what M2's slippage/impact models will quantify properly.")
    print(f"In M1 it's free — but it's already non-zero because of overnight moves.")


if __name__ == "__main__":
    main()