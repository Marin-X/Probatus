"""Manual sanity check for the data handler.

Streams the first 5 bars of AAPL Q1 2024, prints what get_latest_bars()
returns at each step. The history grows by one bar each tick, never
showing future data — that's the no-look-ahead property in action.
"""
from backtester.data import YahooDataHandler


def main():
    data = YahooDataHandler(["AAPL"], start="2024-01-01", end="2024-03-31")
    print(f"Symbols: {data.symbols}")
    print(f"Total bars in timeline: ~{len([1 for _ in range(1000) if data.update() is not None])}")
    # Re-create because the line above exhausted the handler:
    data = YahooDataHandler(["AAPL"], start="2024-01-01", end="2024-03-31")

    for i in range(5):
        event = data.update()
        if event is None:
            break
        bars = data.get_latest_bars("AAPL", n=3)
        print(f"\nTick {i}: current_time = {event.timestamp.date()}")
        print(f"  get_latest_bars(n=3) returned {len(bars)} row(s):")
        print(bars[["Open", "Close"]].round(2).to_string())


if __name__ == "__main__":
    main()