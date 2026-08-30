"""Streamlit dashboard for Indian-market technical-strategy backtesting."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (
    STRATEGIES,
    StrategyParameters,
    evaluate_strategy,
    get_bse_history,
    get_nse_history,
    load_bse500_constituents,
    load_nifty500_constituents,
    metrics_for_display,
)


st.set_page_config(
    page_title="Trading Strategy — MA, RSI & Bollinger",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_history(market: str, identifier: str, start: date, end: date) -> pd.DataFrame:
    """Use the exchange-specific identifier and price-data route."""

    if market == "NSE":
        return get_nse_history(identifier, start, end)
    return get_bse_history(identifier, start, end)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_constituent_universe(market: str) -> tuple[pd.DataFrame, str]:
    """Cache live exchange universes while retaining complete packaged fallbacks."""

    return load_nifty500_constituents() if market == "NSE" else load_bse500_constituents()


def build_price_chart(frame: pd.DataFrame, display_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["close"], name="Adjusted close", line={"color": "#3b82f6", "width": 2}))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["fast_ma"], name="Fast MA", line={"color": "#f59e0b"}))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["slow_ma"], name="Slow MA", line={"color": "#a855f7"}))
    buys = frame[frame["final_signal"] == 1]
    sells = frame[frame["final_signal"] == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["close"], name="Final BUY", mode="markers", marker={"symbol": "triangle-up", "size": 11, "color": "#16a34a"}))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["close"], name="Final SELL", mode="markers", marker={"symbol": "triangle-down", "size": 11, "color": "#dc2626"}))
    fig.update_layout(title=f"{display_name} price, moving averages & final signals", height=460, margin={"l": 12, "r": 12, "t": 48, "b": 12}, legend={"orientation": "h", "y": 1.08})
    fig.update_yaxes(title="₹ price")
    return fig


def build_rsi_chart(frame: pd.DataFrame, params: StrategyParameters) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["rsi"], name="RSI", line={"color": "#8b5cf6", "width": 2}))
    fig.add_hline(y=params.rsi_lower, line_dash="dash", line_color="#16a34a", annotation_text=f"Buy zone {params.rsi_lower:g}")
    fig.add_hline(y=params.rsi_upper, line_dash="dash", line_color="#dc2626", annotation_text=f"Sell zone {params.rsi_upper:g}")
    fig.update_layout(title="Relative Strength Index", height=300, margin={"l": 12, "r": 12, "t": 48, "b": 12}, showlegend=False)
    fig.update_yaxes(range=[0, 100])
    return fig


def build_bollinger_chart(frame: pd.DataFrame, display_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["bb_upper"], name="Upper band", line={"color": "#94a3b8", "dash": "dot"}))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["bb_lower"], name="Lower band", line={"color": "#94a3b8", "dash": "dot"}, fill="tonexty", fillcolor="rgba(148,163,184,0.14)"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["bb_middle"], name="Middle band", line={"color": "#64748b"}))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["close"], name=display_name, line={"color": "#2563eb", "width": 2}))
    fig.update_layout(title="Bollinger Bands", height=350, margin={"l": 12, "r": 12, "t": 48, "b": 12}, legend={"orientation": "h", "y": 1.08})
    fig.update_yaxes(title="₹ price")
    return fig


def build_equity_chart(frame: pd.DataFrame, display_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["strategy_equity"], name="Strategy", line={"color": "#16a34a", "width": 2.5}))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["buy_hold_equity"], name="Buy & hold", line={"color": "#64748b", "dash": "dash"}))
    fig.update_layout(title=f"{display_name} — growth of ₹1", height=380, margin={"l": 12, "r": 12, "t": 48, "b": 12}, legend={"orientation": "h", "y": 1.08})
    fig.update_yaxes(tickprefix="₹")
    return fig


def interpretation(display_name: str, strategy: str, metrics: dict[str, float | int], frame: pd.DataFrame) -> str:
    annual = metrics["annualized_return"]
    buy_hold = metrics["buy_hold_return"]
    strategy_return = metrics["cumulative_return"]
    drawdown = metrics["max_drawdown"]
    active = "outperformed" if strategy_return > buy_hold else "trailed"
    trend = "bullish" if frame["fast_ma"].iloc[-1] > frame["slow_ma"].iloc[-1] else "bearish"
    signal = frame["decision"].iloc[-1]
    return (
        f"For {display_name}, the {strategy} strategy delivered {annual:.1%} annualized return and "
        f"{active} buy-and-hold over this period ({strategy_return:.1%} versus {buy_hold:.1%}). "
        f"Its maximum drawdown was {drawdown:.1%}. The latest moving-average regime is {trend}; "
        f"the latest executable signal is {signal}."
    )


st.title("Trading Strategy — Moving Average, RSI & Bollinger Bands")
st.caption("Compare two NSE or BSE constituents, test indicator agreement, and review risk-adjusted results. Signals are placed at close and enter on the next trading day.")

with st.sidebar:
    st.header("Backtest controls")
    market = st.selectbox("Exchange universe", ("NSE — Nifty 500", "BSE — BSE 500"))
    market_code = "NSE" if market.startswith("NSE") else "BSE"
    if st.button(f"Refresh {market_code} 500 list", width="stretch"):
        load_constituent_universe.clear()
    with st.spinner(f"Loading the {market_code} 500 stock universe…"):
        constituents, universe_source = load_constituent_universe(market_code)
    identifier_column = "Symbol" if market_code == "NSE" else "Scrip Code"
    constituents["Option"] = constituents["Company Name"] + " · " + market_code + " " + constituents[identifier_column]
    option_to_symbol = dict(zip(constituents["Option"], constituents[identifier_column]))
    option_to_name = dict(zip(constituents["Option"], constituents["Company Name"]))
    stock_options = constituents["Option"].tolist()
    tcs_identifier = "TCS" if market_code == "NSE" else "532540"
    infy_identifier = "INFY" if market_code == "NSE" else "500209"
    tcs_index = next((index for index, label in enumerate(stock_options) if option_to_symbol[label] == tcs_identifier), 0)
    infy_index = next((index for index, label in enumerate(stock_options) if option_to_symbol[label] == infy_identifier), 1)
    st.caption(f"{len(constituents):,} {market_code} 500 constituents · {universe_source}")
    stock_one_label = st.selectbox("Stock 1", stock_options, index=tcs_index, key=f"{market_code}_stock_one")
    stock_two_label = st.selectbox("Stock 2", stock_options, index=infy_index, key=f"{market_code}_stock_two")
    stock_one = option_to_symbol[stock_one_label]
    stock_two = option_to_symbol[stock_two_label]
    if stock_one == stock_two:
        st.warning("Choose two different stocks to compare them.")

    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - timedelta(days=365 * 3), max_value=end_date)
    st.divider()
    strategy_name = st.selectbox("Strategy combination", list(STRATEGIES), index=6)
    st.subheader("Moving Average")
    fast_window = st.number_input("Fast MA days", min_value=2, max_value=200, value=20)
    slow_window = st.number_input("Slow MA days", min_value=3, max_value=400, value=50)
    st.subheader("RSI")
    rsi_window = st.number_input("RSI period", min_value=2, max_value=100, value=14)
    rsi_lower = st.number_input("RSI buy threshold", min_value=1, max_value=49, value=30)
    rsi_upper = st.number_input("RSI sell threshold", min_value=51, max_value=99, value=70)
    st.subheader("Bollinger Bands")
    bb_window = st.number_input("Band window", min_value=2, max_value=200, value=20)
    bb_std = st.number_input("Standard-deviation multiplier", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    run = st.button("Refresh results", type="primary", width="stretch")

params = StrategyParameters(
    fast_window=int(fast_window), slow_window=int(slow_window), rsi_window=int(rsi_window),
    rsi_lower=float(rsi_lower), rsi_upper=float(rsi_upper), bollinger_window=int(bb_window), bollinger_std=float(bb_std),
)

current_selection = (market_code, stock_one, stock_two, strategy_name, params, start_date, end_date)
display_names = {stock_one: option_to_name[stock_one_label], stock_two: option_to_name[stock_two_label]}

# A changed stock or parameter always starts a fresh analysis. This prevents a
# stale TCS result from being displayed after the user selects another company.
if run or st.session_state.get("selection") != current_selection:
    try:
        if start_date >= end_date:
            raise ValueError("Choose an end date after the start date.")
        if stock_one == stock_two:
            raise ValueError("Choose two different stocks to compare them.")
        params.validate()
        with st.spinner(f"Downloading {market_code} data and calculating the strategies…"):
            analyses = {}
            for symbol in (stock_one, stock_two):
                prices = load_history(market_code, symbol, start_date, end_date)
                analyses[symbol] = evaluate_strategy(prices, params, strategy_name)
        st.session_state.analysis = analyses
        st.session_state.selection = current_selection
        st.session_state.display_names = display_names
    except Exception as error:
        st.error(str(error))
        st.stop()

if "analysis" not in st.session_state:
    st.info("Set your date range and settings, then select **Run backtest**.")
    st.stop()

analyses = st.session_state.analysis
saved_market, saved_one, saved_two, saved_strategy, saved_params, saved_start, saved_end = st.session_state.selection
saved_display_names = st.session_state.display_names

comparison_rows = []
for symbol in (saved_one, saved_two):
    _, metrics = analyses[symbol]
    comparison_rows.append({"Stock": saved_display_names[symbol], **metrics_for_display(metrics)})
comparison = pd.DataFrame(comparison_rows).set_index("Stock")

st.subheader(f"Results: {saved_strategy}")
st.caption(f"{saved_market} · {saved_start:%d %b %Y} – {saved_end:%d %b %Y} · Strict agreement: every selected indicator must be in the same BUY or SELL state; otherwise the result is HOLD.")
st.dataframe(comparison, width="stretch")

focus_symbol = st.radio("Inspect", [saved_one, saved_two], format_func=lambda symbol: saved_display_names[symbol], horizontal=True)
focus_frame, focus_metrics = analyses[focus_symbol]
focus_name = saved_display_names[focus_symbol]

metric_columns = st.columns(6)
metric_columns[0].metric("Annualized return", f"{focus_metrics['annualized_return']:.1%}")
metric_columns[1].metric("Cumulative return", f"{focus_metrics['cumulative_return']:.1%}")
metric_columns[2].metric("Sharpe ratio", f"{focus_metrics['sharpe_ratio']:.2f}")
metric_columns[3].metric("Maximum drawdown", f"{focus_metrics['max_drawdown']:.1%}")
metric_columns[4].metric("Buy & hold return", f"{focus_metrics['buy_hold_return']:.1%}")
metric_columns[5].metric("Completed trades", f"{focus_metrics['number_of_trades']}", help="A trade is counted after a buy position has later been closed by a sell signal.")

st.plotly_chart(build_price_chart(focus_frame, focus_name), width="stretch")
left, right = st.columns(2)
with left:
    st.plotly_chart(build_rsi_chart(focus_frame, saved_params), width="stretch")
with right:
    st.plotly_chart(build_bollinger_chart(focus_frame, focus_name), width="stretch")
st.plotly_chart(build_equity_chart(focus_frame, focus_name), width="stretch")

st.subheader("Strategy interpretation")
st.write(interpretation(focus_name, saved_strategy, focus_metrics, focus_frame))

st.subheader("Most recent indicator states and execution events")
signal_view = focus_frame.loc[:, ["close", "ma_signal", "rsi", "rsi_signal", "bb_signal", "consensus", "decision", "position"]].tail(25).copy()
signal_view["ma_signal"] = signal_view["ma_signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
signal_view["rsi_signal"] = signal_view["rsi_signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
signal_view["bb_signal"] = signal_view["bb_signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
signal_view = signal_view.rename(columns={"close": "Close (₹)", "ma_signal": "MA state", "rsi": "RSI", "rsi_signal": "RSI state", "bb_signal": "Bollinger state", "consensus": "Aggregator state", "decision": "Execution event", "position": "Invested"})
st.dataframe(signal_view.style.format({"Close (₹)": "₹{:.2f}", "RSI": "{:.1f}", "Invested": "{:.0f}"}), width="stretch")

st.caption("For educational backtesting only. Historical results do not guarantee future performance; review liquidity, costs, taxes, slippage, and corporate actions before investment decisions.")
