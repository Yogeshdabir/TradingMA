# Trading Strategy — Moving Average, RSI & Bollinger Bands

A local Python dashboard for comparing any two Nifty 500 or BSE 500 Indian-market stocks and backtesting moving-average, RSI, Bollinger Band, and strict multi-indicator strategies. It starts with TCS and Infosys in the chosen exchange universe.

## What it does

- Includes complete Nifty 500 and BSE 500 constituent selectors, each with a packaged snapshot and an in-app refresh from the relevant official index site.
- Retrieves adjusted daily data for the chosen two symbols. NSE uses NSE symbols such as `TCS.NS`; BSE resolves each official BSE scrip code to its BSE-compatible price identifier.
- Lets the user search/select two stocks and an inclusive start/end date.
- Calculates individual Moving Average crossover, RSI, and Bollinger Band signals.
- Tests these combinations: MA, RSI, Bollinger, MA+RSI, MA+Bollinger, RSI+Bollinger, and MA+RSI+Bollinger.
- Uses a strict signal aggregator: all selected indicators must emit BUY for a final BUY, all must emit SELL for a final SELL, otherwise the decision is HOLD.
- Backtests long/cash positions with a one-day signal lag; this avoids using same-day closing information to trade that same close.
- Shows indicator/price charts, buy/sell markers, equity curves, a two-stock scorecard, latest signals, and a plain-language interpretation.
- Calculates annualized return, cumulative return, annualized volatility, Sharpe ratio, maximum drawdown, win rate, completed trades, and buy-and-hold comparison.
- Automatically refreshes the analysis whenever the selected exchange, stocks, dates, or parameters change, so a previous TCS result cannot remain on screen after a different company is selected.

## Quick start

Use Python 3.9 or later.

```bash
cd indian-market-backtester
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Streamlit will open a local browser page. Choose the two stocks, set the dates/parameters, choose a strategy combination, and select **Run backtest**.

## File structure

```text
indian-market-backtester/
├── app.py                 # Streamlit dashboard and Plotly charts
├── engine.py              # Data retrieval, indicators, aggregator, and backtest engine
├── data/                  # Complete NSE and BSE constituent fallbacks
├── pyproject.toml         # Test configuration
├── requirements.txt       # Python dependencies
├── README.md              # Setup and methodology
└── tests/
    └── test_engine.py     # Offline unit tests for indicator and backtest logic
```

Run the checks with:

```bash
pytest -q
```

## Strategy rules

### Moving average crossover

- BUY: fast MA crosses above slow MA.
- SELL: fast MA crosses below slow MA.

Defaults are a 20-day fast MA and a 50-day slow MA.

### RSI

- BUY: RSI crosses up out of the oversold threshold.
- SELL: RSI crosses down from the overbought threshold.

Defaults are a 14-day RSI with thresholds 30 and 70.

### Bollinger Bands

- BUY: price crosses below the lower band.
- SELL: price crosses above the upper band.

The defaults use a 20-day rolling window and two standard deviations.

### Aggregation and execution

For a multi-indicator choice, the dashboard requires exact agreement of the current indicator states. For example, MA+RSI enters when both MA and RSI are in their BUY state; if MA says BUY and RSI says HOLD, the final state is HOLD. It emits a BUY or SELL event only when this consensus first changes, so charts do not repeat markers daily. A final BUY enters a long position; a final SELL exits to cash. Returns use yesterday's position, so no strategy can benefit from same-day information.

## R-script reference

The supplied `pairtrading.R` was used as a workflow reference. Its `quantmod::getSymbols("INFY.NS")`/`getSymbols("TCS.NS")` pattern fetches NSE-suffixed symbols through Yahoo Finance, uses rolling measures, and calls `lag(signal, 1)` before returns are calculated. This dashboard uses `yfinance` for the equivalent Python source layer and the same next-day execution safeguard. It implements the requested technical-indicator dashboard rather than its pair-trading ratio/spread strategy.

## Data and limitations

The Nifty 500 universe is sourced from NSE Indices, and the BSE 500 universe is sourced from BSE Indices. The app uses a public Yahoo Finance interface for daily price history, with BSE's official scrip data used to resolve BSE identifiers. This is appropriate for research and teaching. A live trading deployment must replace the public-history adapter with a licensed market-data provider and add brokerage, STT, exchange fees, GST, stamp duty, slippage, liquidity constraints, survivorship handling, corporate-action review, identity controls, auditing, and order-execution safeguards. The results are educational and not investment advice.

Full constituent snapshots are included so stock selection continues working if a public endpoint is unavailable. Both indices are periodically rebalanced, so a live constituent count can change around index reviews and corporate actions.
