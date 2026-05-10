# Probatus

> *Latin: "tested, proven."*

**Probatus** is a Python event-driven backtesting engine for systematic trading strategies. It tests trading signals against realistic execution costs, surfaces overfitting via walk-forward analysis, and enables head-to-head comparison of fundamentally different trading philosophies — trend following, mean reversion, breakout, exhaustion reversal, and a vote-based ensemble.

**→ [Live demo](https://probatus.streamlit.app)** *(deployed via Streamlit Community Cloud)*

---

## Why this exists

Most "backtest" projects on GitHub answer the wrong question. They report gross returns on the in-sample-optimal parameters and call it a day. That number is a mirage: the result of overfit hyperparameters tested without any cost model on the data they were optimized against.

Probatus is built around the four questions that actually decide whether a strategy is real:

1. **Does it survive realistic execution costs?** Per-share IBKR commissions, percent fees (Coinbase), bid-ask half-spread, and Almgren-Chriss square-root market impact (η · σ · √(Q/V) · price).
2. **Is the parameter choice robust, or did one cell of the heatmap win by luck?** Parameter sweep produces a Sharpe heatmap; if a contiguous green region appears, the strategy generalizes. If only one cell is green, it's overfit.
3. **What's the unbiased Sharpe?** Walk-forward analysis: optimize on a rolling 3-year train window, test on the next year. The OOS Sharpe is the only number that isn't an in-sample fantasy.
4. **Does combining philosophies help?** Five different trading paradigms are implemented in the same engine, plus a vote-based ensemble that goes long only when ≥2 of 3 agree.

---

## How it differs from QuantFolio

This is the *complement* to my [QuantFolio](https://github.com/Marin-X/QuantFolio) project, not a duplicate. They answer different questions:

|                     | QuantFolio                                         | Probatus                                              |
|---------------------|----------------------------------------------------|-------------------------------------------------------|
| **Question**        | "Given signals, what's the optimal allocation?"    | "Does this signal make money after realistic costs?"  |
| **Decision unit**   | Asset weights                                      | Trade entry/exit logic                                |
| **Risk view**       | Portfolio-level VaR / CVaR                         | Strategy drawdown, walk-forward degradation           |
| **Diversification** | Across assets                                      | Across philosophies (ensemble)                        |

QuantFolio allocates capital across an existing universe. Probatus decides whether a signal-generating strategy belongs in that universe in the first place.

---

## Strategies

### 1. SMA Crossover — trend following
Goes long when the fast simple moving average crosses above the slow SMA; exits on the reverse cross. The simplest expression of trend following — captures sustained directional moves.

### 2. Bollinger Mean Reversion — mean reversion (statistical)
Goes long when price closes below the lower band (mean − 2σ); exits when price returns to the moving-average mean. The bet is that extreme deviations revert. High win rate, small wins.

### 3. Donchian Breakout — breakout / momentum
Goes long when price breaks above the previous N-bar high; exits when it falls below the previous M-bar low. The original Turtle-trader strategy, foundational to momentum trading.

### 4. Leledc Exhaustion — exhaustion reversal *(Pine Script port)*
Translation of [InSilico's "LeveLeledc" TradingView indicator](https://www.tradingview.com/script/2rZDPyaC-Leledc-Exhaustion-Bar/). Tracks two counters per bar — `bindex` (incremented when close > close[4]) and `sindex` (incremented when close < close[4]). A bullish exhaustion fires when **all three** of these stack on the same bar:

- `sindex > bars` (extended bearish run)
- `close > open` (today reverses bullish)
- `low ≤ rolling-low(length)` (at a new swing low)

Three independent confirmations stacked. Distinct from Bollinger mean reversion: rather than statistical bands, Leledc uses momentum exhaustion + structural confirmation (swing extreme) + single-bar reversal.

### 5. Ensemble — vote-based combination
Goes long only when at least 2 of the 3 base sub-strategies (SMA, Bollinger, Donchian) are simultaneously long. Tests whether *philosophical diversification* produces a smoother equity curve — the strategy-level analogue of asset-level diversification.

---

## Key findings

**Test setup:** AAPL / MSFT / SPY, 2018-01-01 → 2024-12-31, $100k initial cash, IBKR Pro commission + 2bp half-spread, 30% allocation per signal.

| Strategy                        | CAGR  | Sharpe | Calmar | Max DD  | Trades | Win rate |
|---------------------------------|-------|--------|--------|---------|--------|----------|
| SMA Crossover (20/50)           | 11.6% | 0.83   | 0.41   | -28.0%  | 53     | 56.0%    |
| Bollinger Mean Rev (20, 2.0σ)   |  6.4% | 0.53   | 0.30   | -21.7%  | 86     | 76.7%    |
| Donchian Breakout (20/10)       | 11.2% | **0.98** | 0.71  | -15.8%  | 89     | 55.2%    |
| **Ensemble (2-of-3 vote)**      | 11.1% | 0.92   | **0.72** | **-15.4%** | 114 | 54.5% |

**Donchian Breakout has the highest Sharpe** in this period (0.98), and is the only one of the four to beat SPY's Sharpe (0.76) on a risk-adjusted basis.

**The ensemble has the smallest max drawdown (-15.4%) and highest Calmar ratio (0.72)** of any individual strategy, with a CAGR within 0.5pp of the leaders. It does not beat Donchian on raw Sharpe — but philosophical diversification produces the smoothest equity curve, the same effect uncorrelated *assets* have on portfolio volatility achieved at the *strategy* level.

**Bollinger mean reversion has the highest win rate (76.7%) but the lowest CAGR.** Classic mean-reversion behavior in a trending market — wins often, but each win is small, and it sits in cash through the major moves. Would shine in sideways markets.

### Walk-forward (4 folds, 3-year train / 1-year test, SMA crossover)

| Metric                   | Value  |
|--------------------------|--------|
| Avg in-sample Sharpe     | +1.21  |
| Avg out-of-sample Sharpe | +0.60  |
| OOS degradation          | -0.61  |

The OOS Sharpe is roughly **half** the in-sample Sharpe. This is the unbiased estimate of forward performance — and the only number you should use to size capital. Three of four folds had positive OOS Sharpe; the 2022 bear-market fold catastrophically failed (IS Sharpe +1.63 → **OOS Sharpe -1.80** with `fast=10, slow=200`), surfacing the regime risk inherent in long-only trend-following without short selling or volatility filters.

The whole point of building this is that the OOS catastrophe shows up *before* you trade the strategy live, not after.

---

## Architecture

Six-stage event-driven loop, executed once per bar:

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│   Data   │ → │ Strategy │ → │Portfolio │ → │Execution │ → │  Fills   │
│  Handler │   │          │   │          │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
   bars         signals        orders         queued         cash &
                                              (T+1 OPEN)    positions
```

**Look-ahead protection.** Orders queued at the close of bar T fill at the OPEN of bar T+1. The DataHandler enforces this by exposing only `df[df.index <= cutoff]` to strategies; the executor reaches forward by exactly one bar. Look-ahead bugs are the most common silent error in backtesting — this architecture makes them physically impossible.

**Pluggable cost models.** `ZeroCommission`, `PerShareCommission` (IBKR Pro: $0.0035/share, $0.35 min, 1% cap), `PercentCommission` (Coinbase 0.1%), `HalfSpreadSlippage`, `AlmgrenChrissImpact` for size-dependent slippage. Strategy code never knows which cost model is active.

**Stateful per-symbol strategies.** Each strategy maintains its own internal state (e.g. Leledc's bindex/sindex counters, SMA's invested flag). The engine guarantees `on_market(event)` is called once per bar in chronological order.

---

## Features

- **5 interactive Streamlit tabs:** single backtest · trade visualization · parameter sweep · walk-forward · strategy comparison
- **Trade visualization:** candlestick chart with strategy indicators overlaid, BUY/SELL markers at every fill, entry-exit lines colored green (winning trade) or red (losing trade), drag-to-pan / scroll-to-zoom interactivity
- **Parameter sweep heatmaps:** Sharpe and CAGR over (param_a, param_b) grids — visualizes the parameter robustness landscape
- **Walk-forward bar chart:** in-sample vs out-of-sample Sharpe per fold, exposing optimization decay
- **SPY benchmark overlay** with alpha, beta, correlation, and information ratio in single-backtest mode
- **Pluggable cost models** that any strategy can be tested against without code changes

---

## Tech stack

Python 3.12 · pandas · NumPy · yfinance · Streamlit · Plotly. ~2,000 LOC across 9 modules in the `backtester/` package.

```
probatus/
├── backtester/
│   ├── __init__.py        # Public API
│   ├── events.py          # MarketEvent, SignalEvent, OrderEvent, FillEvent
│   ├── data.py            # DataHandler ABC + Yahoo / in-memory implementations
│   ├── strategy.py        # 4 strategies + ensemble (Strategy ABC)
│   ├── portfolio.py       # FixedFractionPortfolio, anti-pyramid guard
│   ├── execution.py       # NextBarOpenExecutor (T+1 OPEN fills)
│   ├── costs.py           # Commission + slippage + Almgren-Chriss impact
│   ├── analytics.py       # Metrics, trade reconstruction, SPY benchmark
│   ├── engine.py          # BacktestEngine event loop
│   └── robustness.py      # parameter_sweep + walk_forward
├── assets/                # Frog logo + favicon
├── .streamlit/            # Theme config
├── app.py                 # Streamlit frontend
└── requirements.txt
```

---

## Quick start

```bash
git clone https://github.com/Marin-X/Probatus.git
cd Probatus
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Browser opens at `http://localhost:8501`. Default backtest runs SMA crossover on AAPL / MSFT / SPY from 2018-2024 with IBKR Pro commission + 2bp slippage. Use the sidebar to switch strategies, edit symbols, or adjust costs.

---

## Author

**Marin Xhemollari** · [marinxhemollari.com](https://marinxhemollari.com) · [LinkedIn](https://linkedin.com/in/marin-xhemollari)

Incoming University of Michigan LSA Data Science (Fall 2026 transfer) · Goal: quantitative trading + Master's in Financial Engineering.
