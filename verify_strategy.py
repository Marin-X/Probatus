"""Sanity check: stream AAPL 2024 daily bars through SMACrossover,
print every signal it emits.
"""
from backtester.data import YahooDataHandler
from backtester.strategy import SMACrossover


def main():
    data = YahooDataHandler(["AAPL"], start="2024-01-01", end="2024-12-31")
    strategy = SMACrossover(data, fast=20, slow=50)

    n = 0
    while True:
        event = data.update()
        if event is None:
            break
        for sig in strategy.on_market(event):
            n += 1
            print(f"{sig.timestamp.date()}  {sig.direction:5s} {sig.symbol}  "
                  f"(strength={sig.strength})")

    print(f"\n{n} signal(s) emitted over the year")


if __name__ == "__main__":
    main()