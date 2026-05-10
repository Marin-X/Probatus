"""Probatus — Streamlit frontend for the multi-strategy event-driven backtester.

Tabs:
    1. Single backtest        - equity, metrics, trade stats, SPY benchmark
    2. Trade visualization    - candlestick + indicators + entry/exit markers
    3. Parameter sweep        - heatmaps over (param_a, param_b) grid
    4. Walk-forward           - rolling train/test optimization
    5. Strategy comparison    - five strategies head-to-head (incl. ensemble)
"""
import base64
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtester import (
    BacktestEngine,
    BollingerMeanReversion,
    DonchianBreakout,
    EnsembleStrategy,
    FixedFractionPortfolio,
    HalfSpreadSlippage,
    InMemoryDataHandler,
    LeledcExhaustion,
    NextBarOpenExecutor,
    NoSlippage,
    PercentCommission,
    PerShareCommission,
    SMACrossover,
    YahooDataHandler,
    ZeroCommission,
    parameter_sweep,
    walk_forward,
)
from backtester.analytics import (
    compare_to_benchmark,
    compute_metrics,
    compute_trade_stats,
    reconstruct_trades,
)


st.set_page_config(
    page_title="Probatus — Marin Xhemollari",
    page_icon="assets/frog-favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)


EMERALD = "#3ECB7A"
AMBER = "#F5D778"
GRAY = "#888888"
TEXT = "#E8F0EA"
RED = "#EF5350"
PALETTE = ["#3ECB7A", "#F5D778", "#5DADE2", "#E67E22", "#A569BD", "#E74C3C"]


# --- Brand styling: match marinxhemollari.com (Georgia / Calibri / Consolas) ---

st.markdown("""
<style>
html, body, [class*="css"], [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], .stMarkdown, .stTextInput, .stSelectbox,
.stButton button, .stMetric, p, span, div, label {
    font-family: 'Calibri', 'Carlito', 'Segoe UI', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Georgia', 'Times New Roman', serif !important;
    color: #E8F0EA;
}
code, pre, [data-testid="stCodeBlock"], .stCode, .language-text {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #3ECB7A !important;
}
[data-testid="stHeader"] {
    background-color: #0D1210;
}
</style>
""", unsafe_allow_html=True)


def load_svg_base64(path: str) -> str:
    """Load an SVG file and return its base64-encoded data URI string."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return ""


_FROG_B64 = load_svg_base64("assets/frog-logo.svg")
_FROG_SRC = f"data:image/svg+xml;base64,{_FROG_B64}" if _FROG_B64 else ""


# --- Strategy registry (sidebar) --------------------------------------------

STRATEGIES = {
    "SMA Crossover":            "sma",
    "Bollinger Mean Reversion": "bollinger",
    "Donchian Breakout":        "donchian",
    "Leledc Exhaustion":        "leledc",
}


def make_strategy(name: str, data, params: dict):
    if name == "sma":
        return SMACrossover(data, fast=params["fast"], slow=params["slow"])
    if name == "bollinger":
        return BollingerMeanReversion(
            data, window=params["window"], num_std=params["num_std"],
        )
    if name == "donchian":
        return DonchianBreakout(
            data, channel=params["channel"],
            exit_channel=params["exit_channel"],
        )
    if name == "leledc":
        return LeledcExhaustion(
            data, length=params["length"], bars=params["bars"],
        )
    if name == "ensemble":
        sub = [
            SMACrossover(data, fast=20, slow=50),
            BollingerMeanReversion(data, window=20, num_std=2.0),
            DonchianBreakout(data, channel=20, exit_channel=10),
        ]
        return EnsembleStrategy(data, sub, min_votes=params.get("min_votes", 2))
    raise ValueError(f"Unknown strategy: {name}")


def strategy_factory(name: str):
    if name == "sma":
        return lambda data, fast, slow: SMACrossover(data, fast=fast, slow=slow)
    if name == "bollinger":
        return lambda data, window, num_std: BollingerMeanReversion(
            data, window=window, num_std=num_std)
    if name == "donchian":
        return lambda data, channel, exit_channel: DonchianBreakout(
            data, channel=channel, exit_channel=exit_channel)
    if name == "leledc":
        return lambda data, length, bars: LeledcExhaustion(
            data, length=length, bars=bars)
    raise ValueError(f"Unknown strategy: {name}")


def strategy_validity(name: str):
    if name == "sma":
        return lambda fast, slow: fast < slow
    if name == "donchian":
        return lambda channel, exit_channel: exit_channel <= channel
    return None


# --- Cached data download ---------------------------------------------------

@st.cache_data(show_spinner=False)
def download_data(symbols: tuple, start: str, end: str) -> dict:
    handler = YahooDataHandler(list(symbols), start=start, end=end)
    return {sym: handler.get_symbol_history(sym) for sym in symbols}


def make_cost_models(commission_choice: str, slippage_bps: float):
    comm = {
        "Zero":           ZeroCommission(),
        "IBKR Pro":       PerShareCommission(),
        "Coinbase 0.1%":  PercentCommission(0.001),
    }[commission_choice]
    slip = HalfSpreadSlippage(half_spread_bps=slippage_bps) if slippage_bps > 0 else NoSlippage()
    return comm, slip


def styled_layout(title: str = "", height: int = 500) -> dict:
    return dict(
        title=title, height=height, hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
        margin=dict(l=60, r=20, t=50, b=40),
    )


def run_single_backtest(strategy_key, params, symbols, start, end,
                        initial_cash, allocation, comm, slip):
    raw = download_data(symbols, start, end)
    data = InMemoryDataHandler({s: df.copy() for s, df in raw.items()})
    strategy = make_strategy(strategy_key, data, params)
    portfolio = FixedFractionPortfolio(
        data, initial_cash=initial_cash, allocation_fraction=allocation,
    )
    execution = NextBarOpenExecutor(
        data, commission_model=comm, slippage_model=slip,
    )
    engine = BacktestEngine(data, strategy, portfolio, execution)
    engine.run()
    return raw, portfolio


# --- Sidebar -----------------------------------------------------------------

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem;">
    <img src="{_FROG_SRC}" style="width: 32px; height: 32px;" alt="logo"/>
    <span style="font-family: Georgia, serif; font-size: 1.3rem;
                 color: #E8F0EA;">Probatus</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.header("Configuration")

strategy_label = st.sidebar.selectbox(
    "Strategy", list(STRATEGIES.keys()), index=0,
)
strategy_key = STRATEGIES[strategy_label]

symbols_input = st.sidebar.text_input(
    "Symbols (comma-separated)", value="AAPL,MSFT,SPY",
)
symbols = tuple(s.strip().upper() for s in symbols_input.split(",") if s.strip())

c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("Start", value=pd.Timestamp("2018-01-01"))
end_date = c2.date_input("End", value=pd.Timestamp("2024-12-31"))

st.sidebar.subheader("Strategy parameters")
strategy_params: dict = {}
if strategy_key == "sma":
    strategy_params["fast"] = st.sidebar.slider("Fast SMA", 5, 100, value=20)
    strategy_params["slow"] = st.sidebar.slider("Slow SMA", 20, 300, value=50)
elif strategy_key == "bollinger":
    strategy_params["window"] = st.sidebar.slider("Window", 5, 100, value=20)
    strategy_params["num_std"] = st.sidebar.slider(
        "Standard deviations", 0.5, 4.0, value=2.0, step=0.25,
    )
elif strategy_key == "donchian":
    strategy_params["channel"] = st.sidebar.slider("Entry channel", 5, 100, value=20)
    strategy_params["exit_channel"] = st.sidebar.slider("Exit channel", 5, 100, value=10)
elif strategy_key == "leledc":
    strategy_params["length"] = st.sidebar.slider(
        "Swing length", 10, 100, value=40,
        help="Lookback window for swing high/low detection (Pine: 'length')",
    )
    strategy_params["bars"] = st.sidebar.slider(
        "Exhaustion bars", 3, 30, value=10,
        help="Min consecutive same-direction closes for exhaustion (Pine: 'bars')",
    )

st.sidebar.subheader("Portfolio")
initial_cash = st.sidebar.number_input("Initial cash ($)", value=100_000, step=10_000)
allocation = st.sidebar.slider("Allocation per signal (%)", 5, 100, value=30) / 100

st.sidebar.subheader("Costs")
commission_choice = st.sidebar.selectbox(
    "Commission model", ["Zero", "IBKR Pro", "Coinbase 0.1%"], index=1,
)
slippage_bps = st.sidebar.slider("Half-spread (bps)", 0.0, 20.0, value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Probatus** — multi-strategy event-driven backtester. Trend following, "
    "mean reversion, breakout, exhaustion, and ensemble — five philosophies, "
    "one engine."
)


# --- Main --------------------------------------------------------------------

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 1.2rem; margin-bottom: 0.3rem;">
    <img src="{_FROG_SRC}" style="width: 72px; height: 72px; flex-shrink: 0;" alt="logo"/>
    <h1 style="font-family: Georgia, serif; font-size: 3.2rem; margin: 0;
               color: #E8F0EA;">Probatus</h1>
</div>
<p style="color: #888888; font-family: Calibri, Carlito, sans-serif;
          font-size: 1rem; margin-top: 0; margin-bottom: 1.5rem;">
    Event-driven multi-strategy backtester · realistic execution costs ·
    walk-forward robustness · trade-level analytics
</p>
""", unsafe_allow_html=True)

tab_backtest, tab_viz, tab_sweep, tab_wf, tab_compare = st.tabs([
    "Single backtest", "Trade visualization", "Parameter sweep",
    "Walk-forward", "Strategy comparison",
])


# === Tab 1: single backtest ==================================================

with tab_backtest:
    if not symbols:
        st.error("Enter at least one symbol.")
    else:
        if st.button("Run backtest", type="primary", key="run_single"):
            with st.spinner(f"Backtesting {strategy_label} on {list(symbols)}..."):
                comm, slip = make_cost_models(commission_choice, slippage_bps)
                try:
                    raw, portfolio = run_single_backtest(
                        strategy_key, strategy_params, symbols,
                        str(start_date), str(end_date),
                        initial_cash, allocation, comm, slip,
                    )
                except ValueError as e:
                    st.error(f"Parameter error: {e}")
                    st.stop()

                curve = portfolio.equity_curve()
                metrics = compute_metrics(curve)
                trades = reconstruct_trades(portfolio.fills)
                stats = compute_trade_stats(trades)
                bench = compare_to_benchmark(curve, raw["SPY"]["Close"], "SPY") \
                    if "SPY" in symbols else None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total return", f"{metrics.total_return:.1%}")
            c2.metric("CAGR", f"{metrics.cagr:.1%}")
            c3.metric("Sharpe", f"{metrics.sharpe_ratio:.2f}")
            c4.metric("Max drawdown", f"{metrics.max_drawdown:.1%}")

            peak = curve["equity"].cummax()
            drawdown = (curve["equity"] - peak) / peak
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                row_heights=[0.72, 0.28],
            )
            fig.add_trace(go.Scatter(
                x=curve.index, y=curve["equity"],
                line=dict(color=EMERALD, width=2),
                name=strategy_label,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>",
            ), row=1, col=1)
            if bench is not None:
                bench_eq = raw["SPY"]["Close"].reindex(curve.index, method="ffill")
                bench_eq = initial_cash * bench_eq / bench_eq.iloc[0]
                fig.add_trace(go.Scatter(
                    x=bench_eq.index, y=bench_eq.values,
                    line=dict(color=GRAY, width=1.5, dash="dash"),
                    name="SPY buy & hold",
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>",
                ), row=1, col=1)
            fig.add_hline(y=initial_cash, line_dash="dot",
                          line_color="gray", opacity=0.5, row=1, col=1)
            fig.add_trace(go.Scatter(
                x=drawdown.index, y=drawdown.values,
                fill="tozeroy", fillcolor="rgba(245,215,120,0.3)",
                line=dict(color=AMBER, width=1),
                name="Drawdown", showlegend=False,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.1%}<extra></extra>",
            ), row=2, col=1)
            fig.update_yaxes(title_text="Equity ($)", row=1, col=1,
                             gridcolor="rgba(255,255,255,0.1)")
            fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1,
                             gridcolor="rgba(255,255,255,0.1)")
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)", row=2, col=1)
            fig.update_layout(**styled_layout(height=620))
            st.plotly_chart(fig, use_container_width=True)

            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("Performance")
                st.code(str(metrics), language="text")
            with c_right:
                st.subheader("Trade statistics")
                st.code(str(stats), language="text")
            if bench is not None:
                st.subheader("vs SPY benchmark")
                st.code(str(bench), language="text")


# === Tab 2: trade visualization =============================================

with tab_viz:
    st.markdown(
        "Watch the strategy trade on a real candlestick chart. "
        "Indicators are overlaid, BUYs are green ▲ and SELLs are red ▼, "
        "and each completed trade is connected by a colored line "
        "(green = winning trade, red = losing trade). "
        "**Drag** the chart to pan, use the **rangeslider at the bottom** to zoom into a period, "
        "**scroll** to zoom in/out."
    )

    if not symbols:
        st.error("Enter symbols in the sidebar.")
    else:
        col_a, col_b = st.columns([2, 1])
        viz_symbol = col_a.selectbox(
            "Symbol to visualize", list(symbols), key="viz_sym",
        )
        col_b.markdown("&nbsp;")
        run_viz = col_b.button(
            "Run + visualize", type="primary", key="run_viz",
        )

        if run_viz:
            with st.spinner(f"Backtesting {strategy_label}..."):
                comm, slip = make_cost_models(commission_choice, slippage_bps)
                try:
                    raw, portfolio = run_single_backtest(
                        strategy_key, strategy_params, symbols,
                        str(start_date), str(end_date),
                        initial_cash, allocation, comm, slip,
                    )
                except ValueError as e:
                    st.error(f"Parameter error: {e}")
                    st.stop()

            df = raw[viz_symbol].copy()
            df = df[(df.index >= pd.Timestamp(start_date)) &
                    (df.index <= pd.Timestamp(end_date))]
            if df.empty:
                st.warning(f"No data for {viz_symbol} in the selected range.")
                st.stop()

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"],
                increasing_line_color=EMERALD, increasing_fillcolor=EMERALD,
                decreasing_line_color=RED, decreasing_fillcolor=RED,
                name="Price", showlegend=False,
            ))

            if strategy_key == "sma":
                f = strategy_params["fast"]; s = strategy_params["slow"]
                fast_ma = df["Close"].rolling(f).mean()
                slow_ma = df["Close"].rolling(s).mean()
                fig.add_trace(go.Scatter(
                    x=df.index, y=fast_ma, name=f"SMA {f}",
                    line=dict(color=EMERALD, width=1.4),
                    hovertemplate="SMA " + str(f) + ": $%{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=slow_ma, name=f"SMA {s}",
                    line=dict(color=AMBER, width=1.4),
                    hovertemplate="SMA " + str(s) + ": $%{y:.2f}<extra></extra>",
                ))
            elif strategy_key == "bollinger":
                w = strategy_params["window"]; ns = strategy_params["num_std"]
                middle = df["Close"].rolling(w).mean()
                std = df["Close"].rolling(w).std(ddof=0)
                upper = middle + ns * std
                lower = middle - ns * std
                fig.add_trace(go.Scatter(
                    x=df.index, y=middle, name=f"SMA {w}",
                    line=dict(color=GRAY, width=1.0),
                    hovertemplate="Middle: $%{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=upper, name=f"+{ns}σ",
                    line=dict(color=AMBER, width=1.0, dash="dot"),
                    hovertemplate="Upper: $%{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=lower, name=f"-{ns}σ",
                    line=dict(color=AMBER, width=1.0, dash="dot"),
                    hovertemplate="Lower: $%{y:.2f}<extra></extra>",
                ))
            elif strategy_key == "donchian":
                ch = strategy_params["channel"]; ex = strategy_params["exit_channel"]
                upper = df["High"].rolling(ch).max().shift(1)
                lower = df["Low"].rolling(ex).min().shift(1)
                fig.add_trace(go.Scatter(
                    x=df.index, y=upper, name=f"{ch}-bar high",
                    line=dict(color=AMBER, width=1.2),
                    hovertemplate="Upper: $%{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=lower, name=f"{ex}-bar low",
                    line=dict(color=GRAY, width=1.2),
                    hovertemplate="Lower: $%{y:.2f}<extra></extra>",
                ))
            elif strategy_key == "leledc":
                # Show the rolling swing-high / swing-low envelope used in
                # the exhaustion detection. Doesn't change between bars
                # like the levels do, but it's the input the rule scans.
                ln = strategy_params["length"]
                roll_hi = df["High"].rolling(ln).max()
                roll_lo = df["Low"].rolling(ln).min()
                fig.add_trace(go.Scatter(
                    x=df.index, y=roll_hi, name=f"{ln}-bar swing high",
                    line=dict(color=AMBER, width=1.0, dash="dot"),
                    hovertemplate="Swing hi: $%{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=roll_lo, name=f"{ln}-bar swing low",
                    line=dict(color=GRAY, width=1.0, dash="dot"),
                    hovertemplate="Swing lo: $%{y:.2f}<extra></extra>",
                ))

            trades = reconstruct_trades(portfolio.fills)
            sym_trades = [t for t in trades
                          if t.symbol == viz_symbol and not t.is_open]
            for t in sym_trades:
                color = (
                    "rgba(62, 203, 122, 0.7)" if t.pnl > 0
                    else "rgba(239, 83, 80, 0.7)"
                )
                fig.add_trace(go.Scatter(
                    x=[t.entry_time, t.exit_time],
                    y=[t.entry_price, t.exit_price],
                    mode="lines", showlegend=False,
                    line=dict(color=color, width=2),
                    hoverinfo="skip",
                ))

            buy_fills = [
                f for f in portfolio.fills
                if f.symbol == viz_symbol and f.direction == "BUY"
            ]
            if buy_fills:
                fig.add_trace(go.Scatter(
                    x=[f.timestamp for f in buy_fills],
                    y=[f.fill_price for f in buy_fills],
                    mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", size=14, color=EMERALD,
                                line=dict(width=1.5, color="white")),
                    hovertemplate=("<b>BUY</b><br>%{x|%Y-%m-%d}<br>"
                                   "$%{y:.2f}<extra></extra>"),
                ))

            sell_fills = [
                f for f in portfolio.fills
                if f.symbol == viz_symbol and f.direction == "SELL"
            ]
            if sell_fills:
                fig.add_trace(go.Scatter(
                    x=[f.timestamp for f in sell_fills],
                    y=[f.fill_price for f in sell_fills],
                    mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", size=14, color=RED,
                                line=dict(width=1.5, color="white")),
                    hovertemplate=("<b>SELL</b><br>%{x|%Y-%m-%d}<br>"
                                   "$%{y:.2f}<extra></extra>"),
                ))

            n_trades = len(sym_trades)
            n_winners = sum(1 for t in sym_trades if t.pnl > 0)
            total_pnl = sum(t.pnl for t in sym_trades)
            best = max((t.pnl for t in sym_trades), default=0.0)
            worst = min((t.pnl for t in sym_trades), default=0.0)

            fig.update_xaxes(rangeslider_visible=False,
                             range=[df.index.min(), df.index.max()],
                             gridcolor="rgba(255,255,255,0.1)")
            fig.update_yaxes(title_text="Price ($)",
                             gridcolor="rgba(255,255,255,0.1)",
                             fixedrange=False)
            fig.update_layout(**styled_layout(
                title=(f"{viz_symbol} · {strategy_label} · "
                       f"{n_trades} trades · "
                       f"{n_winners}/{n_trades} winners · "
                       f"P&L ${total_pnl:+,.0f}"),
                height=720,
            ))
            fig.update_layout(dragmode="pan")
            st.plotly_chart(
                fig, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )

            if sym_trades:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"Trades on {viz_symbol}", n_trades)
                c2.metric(
                    "Win rate",
                    f"{n_winners / n_trades:.1%}" if n_trades else "—",
                )
                c3.metric("Best trade", f"${best:+,.0f}")
                c4.metric("Worst trade", f"${worst:+,.0f}")


# === Tab 3: parameter sweep ==================================================

def get_sweep_grid(strategy_key):
    if strategy_key == "sma":
        return (("10,20,30,50", "50,100,150,200"),
                ("fast", "slow"),
                ("Fast values", "Slow values"))
    if strategy_key == "bollinger":
        return (("10,20,30,50", "1.5,2.0,2.5,3.0"),
                ("window", "num_std"),
                ("Window values", "Num-std values"))
    if strategy_key == "donchian":
        return (("10,20,30,50", "5,10,15,20"),
                ("channel", "exit_channel"),
                ("Channel values", "Exit-channel values"))
    if strategy_key == "leledc":
        return (("20,40,60,80", "5,10,15,20"),
                ("length", "bars"),
                ("Length values", "Bars values"))


with tab_sweep:
    st.markdown(
        f"Run **{strategy_label}** for every combination on the grid. "
        "If a contiguous green region appears in the heatmap, the "
        "strategy is robust. If only one cell is green, it's overfit."
    )
    (def_a, def_b), (key_a, key_b), (lbl_a, lbl_b) = get_sweep_grid(strategy_key)
    c1, c2 = st.columns(2)
    vals_a_str = c1.text_input(f"{lbl_a} (csv)", value=def_a, key="va")
    vals_b_str = c2.text_input(f"{lbl_b} (csv)", value=def_b, key="vb")

    if st.button("Run sweep", type="primary", key="run_sweep"):
        try:
            a_type = float if key_a == "num_std" else int
            b_type = float if key_b == "num_std" else int
            vals_a = [a_type(x) for x in vals_a_str.split(",")]
            vals_b = [b_type(x) for x in vals_b_str.split(",")]
        except ValueError:
            st.error("Invalid values.")
            st.stop()

        comm, slip = make_cost_models(commission_choice, slippage_bps)
        progress = st.progress(0.0, text="Starting sweep...")
        t0 = time.time()

        def cb(current, total, message):
            elapsed = int(time.time() - t0)
            pct = current / total if total > 0 else 0.0
            progress.progress(min(pct, 1.0),
                              text=f"[{current}/{total}] {message}  ·  {elapsed}s")

        sweep = parameter_sweep(
            symbols=list(symbols),
            start=str(start_date), end=str(end_date),
            strategy_factory=strategy_factory(strategy_key),
            param_grid={key_a: vals_a, key_b: vals_b},
            is_valid_combination=strategy_validity(strategy_key),
            initial_cash=initial_cash, allocation_fraction=allocation,
            commission_model=comm, slippage_model=slip,
            verbose=False, progress_callback=cb,
        )
        progress.empty()
        st.success(f"Sweep complete in {int(time.time() - t0)}s.")

        if sweep.empty:
            st.warning("No valid combinations.")
            st.stop()

        st.subheader("Results (sorted by Sharpe)")
        st.dataframe(
            sweep.sort_values("sharpe", ascending=False).style.format({
                "total_return": "{:.1%}", "cagr": "{:.1%}",
                "sharpe": "{:.2f}", "sortino": "{:.2f}",
                "max_dd": "{:.1%}", "calmar": "{:.2f}", "vol": "{:.1%}",
                "num_std": "{:.2f}",
            }),
            height=400,
        )

        try:
            pivot_sharpe = sweep.pivot(index=key_a, columns=key_b, values="sharpe")
            pivot_cagr = sweep.pivot(index=key_a, columns=key_b, values="cagr")
        except Exception:
            st.info("Non-rectangular grid; skipping heatmap.")
            st.stop()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Sharpe ratio", "CAGR"),
            horizontal_spacing=0.12,
        )
        for col_idx, (pivot, fmt) in enumerate([
            (pivot_sharpe, "{:+.2f}"),
            (pivot_cagr, "{:+.1%}"),
        ], start=1):
            text = [
                [fmt.format(v) if not np.isnan(v) else "" for v in row]
                for row in pivot.values
            ]
            fig.add_trace(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale="RdYlGn",
                text=text, texttemplate="%{text}",
                textfont=dict(size=12, color="black"),
                showscale=(col_idx == 1),
                hovertemplate=(f"{key_a}=%{{y}}, {key_b}=%{{x}}<br>"
                               "%{text}<extra></extra>"),
            ), row=1, col=col_idx)
        fig.update_xaxes(title_text=key_b, row=1, col=1)
        fig.update_xaxes(title_text=key_b, row=1, col=2)
        fig.update_yaxes(title_text=key_a, row=1, col=1)
        fig.update_yaxes(title_text=key_a, row=1, col=2)
        fig.update_layout(**styled_layout(height=480))
        st.plotly_chart(fig, use_container_width=True)


# === Tab 4: walk-forward =====================================================

with tab_wf:
    st.markdown(
        f"**Walk-forward** for **{strategy_label}**: optimize on a rolling "
        "train window, test on the next window. The OOS Sharpe is the "
        "*unbiased* estimate. If OOS << IS, in-sample optimization is "
        "overfitting period-specific noise."
    )
    c1, c2 = st.columns(2)
    train_years = c1.slider("Train years", 1.0, 5.0, value=3.0, step=0.5)
    test_years = c2.slider("Test years", 0.5, 3.0, value=1.0, step=0.5)

    (def_a, def_b), (key_a, key_b), (lbl_a, lbl_b) = get_sweep_grid(strategy_key)
    c3, c4 = st.columns(2)
    vals_a_wf = c3.text_input(f"{lbl_a}", value=def_a, key="va_wf")
    vals_b_wf = c4.text_input(f"{lbl_b}", value=def_b, key="vb_wf")

    if st.button("Run walk-forward", type="primary", key="run_wf"):
        try:
            a_type = float if key_a == "num_std" else int
            b_type = float if key_b == "num_std" else int
            vals_a = [a_type(x) for x in vals_a_wf.split(",")]
            vals_b = [b_type(x) for x in vals_b_wf.split(",")]
        except ValueError:
            st.error("Invalid values.")
            st.stop()

        comm, slip = make_cost_models(commission_choice, slippage_bps)
        progress = st.progress(0.0, text="Starting walk-forward...")
        t0 = time.time()

        def wf_cb(current, total, message):
            elapsed = int(time.time() - t0)
            pct = current / total if total > 0 else 0.0
            progress.progress(min(pct, 1.0),
                              text=f"[{current}/{total}] {message}  ·  {elapsed}s")

        folds = walk_forward(
            symbols=list(symbols),
            start=str(start_date), end=str(end_date),
            strategy_factory=strategy_factory(strategy_key),
            param_grid={key_a: vals_a, key_b: vals_b},
            is_valid_combination=strategy_validity(strategy_key),
            train_years=train_years, test_years=test_years,
            initial_cash=initial_cash, allocation_fraction=allocation,
            commission_model=comm, slippage_model=slip,
            verbose=False, progress_callback=wf_cb,
        )
        progress.empty()
        st.success(f"Walk-forward complete in {int(time.time() - t0)}s.")

        if not folds:
            st.warning("No folds produced.")
            st.stop()

        avg_is = float(np.mean([f.train_sharpe for f in folds]))
        avg_oos = float(np.mean([f.test_sharpe for f in folds]))
        degradation = avg_oos - avg_is
        avg_oos_cagr = float(np.mean([f.test_cagr for f in folds]))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Folds", len(folds))
        c2.metric("Avg IS Sharpe", f"{avg_is:+.2f}")
        c3.metric("Avg OOS Sharpe", f"{avg_oos:+.2f}",
                  delta=f"{degradation:+.2f}", delta_color="inverse")
        c4.metric("Avg OOS CAGR", f"{avg_oos_cagr:+.1%}")

        rows = [{
            "Fold": i + 1,
            "Train": f"{f.train_start.date()} → {f.train_end.date()}",
            "Test":  f"{f.test_start.date()} → {f.test_end.date()}",
            "Best params": ", ".join(f"{k}={v}" for k, v in f.best_params.items()),
            "IS Sharpe": f"{f.train_sharpe:+.2f}",
            "OOS Sharpe": f"{f.test_sharpe:+.2f}",
            "OOS CAGR": f"{f.test_cagr:+.1%}",
            "OOS Max DD": f"{f.test_max_dd:.1%}",
        } for i, f in enumerate(folds)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(folds))],
            y=[f.train_sharpe for f in folds],
            name="In-sample Sharpe", marker_color=GRAY,
        ))
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(folds))],
            y=[f.test_sharpe for f in folds],
            name="Out-of-sample Sharpe", marker_color=EMERALD,
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.6)
        fig.update_layout(**styled_layout(
            title="In-sample vs out-of-sample Sharpe by fold",
            height=440,
        ))
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)


# === Tab 5: strategy comparison ==============================================

with tab_compare:
    st.markdown(
        "Run **all five strategies** at default parameters on the same data "
        "and compare their equity curves and risk-adjusted metrics. Each "
        "implements a different philosophy — trend following, mean reversion, "
        "breakout, exhaustion reversal, or a 2-of-3 ensemble — and the "
        "comparison surfaces which regime each one wins in."
    )

    if st.button("Compare all strategies", type="primary", key="run_compare"):
        defaults = [
            ("SMA Crossover (20/50)",         "sma",       {"fast": 20, "slow": 50}),
            ("Bollinger Mean Rev (20, 2.0σ)", "bollinger", {"window": 20, "num_std": 2.0}),
            ("Donchian Breakout (20/10)",     "donchian",  {"channel": 20, "exit_channel": 10}),
            ("Leledc Exhaustion (40/10)",     "leledc",    {"length": 40, "bars": 10}),
            ("Ensemble (2-of-3 vote)",        "ensemble",  {"min_votes": 2}),
        ]
        comm, slip = make_cost_models(commission_choice, slippage_bps)
        progress = st.progress(0.0, text="Starting comparison...")
        t0 = time.time()

        results = []
        raw = None
        for i, (label, key, params) in enumerate(defaults):
            progress.progress(
                (i + 0.1) / len(defaults),
                text=f"[{i+1}/{len(defaults)}] {label}  ·  {int(time.time() - t0)}s",
                )
            try:
                raw, portfolio = run_single_backtest(
                    key, params, symbols,
                    str(start_date), str(end_date),
                    initial_cash, allocation, comm, slip,
                )
            except ValueError as e:
                st.error(f"{label}: {e}")
                continue
            curve = portfolio.equity_curve()
            metrics = compute_metrics(curve)
            trades = reconstruct_trades(portfolio.fills)
            stats = compute_trade_stats(trades)
            results.append({
                "label": label, "curve": curve,
                "metrics": metrics, "stats": stats,
                "trade_count": len(trades),
            })

        progress.empty()
        st.success(f"Comparison complete in {int(time.time() - t0)}s.")

        if not results:
            st.stop()

        fig = go.Figure()
        for i, r in enumerate(results):
            fig.add_trace(go.Scatter(
                x=r["curve"].index, y=r["curve"]["equity"],
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                name=r["label"],
                hovertemplate=("<b>" + r["label"] + "</b><br>%{x|%Y-%m-%d}<br>"
                                                    "$%{y:,.0f}<extra></extra>"),
            ))
        if "SPY" in symbols and raw is not None:
            spy = raw["SPY"]["Close"]
            spy_eq = initial_cash * spy / spy.iloc[0]
            fig.add_trace(go.Scatter(
                x=spy_eq.index, y=spy_eq.values,
                line=dict(color=GRAY, width=1.5, dash="dash"),
                name="SPY buy & hold",
                hovertemplate=("<b>SPY</b><br>%{x|%Y-%m-%d}<br>"
                               "$%{y:,.0f}<extra></extra>"),
            ))
        fig.add_hline(y=initial_cash, line_dash="dot",
                      line_color="gray", opacity=0.5)
        fig.update_yaxes(title_text="Equity ($)")
        fig.update_layout(**styled_layout(
            title="Strategy comparison — five philosophies, same data",
            height=560,
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk-adjusted comparison")
        rows = []
        for r in results:
            m = r["metrics"]
            s = r["stats"]
            rows.append({
                "Strategy": r["label"],
                "Total return": f"{m.total_return:.1%}",
                "CAGR": f"{m.cagr:.1%}",
                "Sharpe": f"{m.sharpe_ratio:.2f}",
                "Sortino": f"{m.sortino_ratio:.2f}",
                "Calmar": f"{m.calmar_ratio:.2f}",
                "Max DD": f"{m.max_drawdown:.1%}",
                "Trades": r["trade_count"],
                "Win rate": f"{s.win_rate:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown(
            "*Trend following* (SMA) wins in clean directional periods. "
            "*Mean reversion* (Bollinger) wins in choppy, range-bound markets. "
            "*Breakout* (Donchian) wins when major regime shifts produce "
            "sustained moves. *Exhaustion reversal* (Leledc, ported from "
            "InSilico's TradingView Pine Script) catches turning points "
            "after extended runs. The *ensemble* (2-of-3 vote across SMA, "
            "Bollinger, Donchian) tests whether philosophical diversification "
            "smooths the equity curve. Run different time periods or symbols "
            "to see which regime dominates."
        )