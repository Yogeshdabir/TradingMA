"""Indicator, signal aggregation, and long-only backtesting utilities.

Signals are created from data available at the daily close and are applied to the
following session's return.  That one-session delay prevents look-ahead bias.
"""

from __future__ import annotations

from io import BytesIO
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


TRADING_DAYS = 252
OFFICIAL_NIFTY500_URLS = (
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
)
PACKAGED_NIFTY500_CSV = Path(__file__).with_name("data") / "nifty500_constituents.csv"
BSE500_CONSTITUENTS_URL = "https://www.bseindices.com/AsiaIndexAPI/api/Codewise_Indices/w?code=17"
BSE_SCRIP_HEADER_URL = "https://api.bseindia.com/BseIndiaAPI/api/getScripHeaderData/w"
PACKAGED_BSE500_JSON = Path(__file__).with_name("data") / "bse500_constituents.json"


@dataclass(frozen=True)
class StrategyParameters:
    """User-controlled settings shared by each supported strategy."""

    fast_window: int = 20
    slow_window: int = 50
    rsi_window: int = 14
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0
    bollinger_window: int = 20
    bollinger_std: float = 2.0

    def validate(self) -> None:
        if self.fast_window >= self.slow_window:
            raise ValueError("The fast moving-average window must be smaller than the slow window.")
        if not 0 < self.rsi_lower < self.rsi_upper < 100:
            raise ValueError("RSI thresholds must satisfy 0 < lower < upper < 100.")
        if self.bollinger_window < 2 or self.bollinger_std <= 0:
            raise ValueError("Bollinger settings must use a window of at least 2 and a positive deviation.")


def _download_yahoo_history(symbol: str, start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DataFrame | None:
    """Fetch and normalize one Yahoo Finance symbol, returning None when absent."""

    import yfinance as yf

    normalized = symbol.strip().upper()
    # yfinance treats `end` as exclusive, so request one additional calendar day.
    history = yf.Ticker(normalized).history(
        start=pd.Timestamp(start).date(),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date(),
        auto_adjust=True,
        actions=False,
    )
    if history.empty:
        return None

    history.index = pd.to_datetime(history.index)
    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    history = history.rename(columns=str.lower)
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(history.columns)
    if missing:
        return None
    return history.loc[:, ["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def get_nse_history(symbol: str, start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DataFrame:
    """Download daily history for an NSE symbol such as ``TCS`` or ``TCS.NS``."""

    normalized = symbol.strip().upper()
    if not normalized.endswith(".NS"):
        normalized = f"{normalized}.NS"
    history = _download_yahoo_history(normalized, start, end)
    if history is None or history.empty:
        raise ValueError(f"No NSE price history was returned for {normalized}. Check the dates and symbol.")
    return history


def _fetch_bse_short_name(scrip_code: str) -> str | None:
    """Get BSE's short trading name for a single BSE scrip code."""

    request = Request(
        f"{BSE_SCRIP_HEADER_URL}?Debtflag=&scripcode={scrip_code}&seriesid=",
        headers={"User-Agent": "Mozilla/5.0 (compatible; NSE-Strategy-Lab/1.0)", "Referer": "https://www.bseindia.com/"},
    )
    try:
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        name = payload.get("Cmpname", {}).get("ShortN", "").strip()
        return name.upper() or None
    except Exception:
        return None


def get_bse_history(scrip_code: str, start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DataFrame:
    """Download daily BSE history using BSE scrip-code and symbol fallbacks.

    Yahoo Finance uses BSE identifiers inconsistently: some shares resolve by
    six-digit scrip code (for example ``500209.BO``), while others resolve by
    BSE's short trading name (for example ``TCS.BO``). Both official BSE
    identifiers are tried before reporting an unavailable series.
    """

    normalized_code = str(scrip_code).strip().zfill(6)
    candidates: list[str] = []
    short_name = _fetch_bse_short_name(normalized_code)
    if short_name:
        candidates.append(f"{short_name}.BO")
    candidates.append(f"{normalized_code}.BO")
    for candidate in dict.fromkeys(candidates):
        history = _download_yahoo_history(candidate, start, end)
        if history is not None and not history.empty:
            return history
    raise ValueError(
        f"No BSE price history was returned for scrip {normalized_code}. "
        "Try a different BSE 500 constituent or a later date range."
    )


def _parse_constituents_csv(raw_csv: bytes) -> pd.DataFrame:
    """Validate and normalize the Nifty 500 constituent CSV."""

    constituents = pd.read_csv(BytesIO(raw_csv), encoding="utf-8-sig")
    constituents.columns = [str(column).strip() for column in constituents.columns]
    required = {"Company Name", "Symbol"}
    missing = required.difference(constituents.columns)
    if missing:
        raise ValueError(f"Constituent file is missing: {', '.join(sorted(missing))}.")
    keep = [column for column in ("Company Name", "Industry", "Symbol", "Series", "ISIN Code") if column in constituents]
    constituents = constituents.loc[:, keep].copy()
    constituents["Symbol"] = constituents["Symbol"].astype(str).str.strip().str.upper()
    constituents["Company Name"] = constituents["Company Name"].astype(str).str.strip()
    constituents = constituents.loc[constituents["Symbol"].ne("") & constituents["Symbol"].ne("NAN")]
    constituents = constituents.drop_duplicates(subset="Symbol").sort_values("Symbol").reset_index(drop=True)
    if len(constituents) < 490:
        raise ValueError(f"Only {len(constituents)} symbols were found; this is not a complete Nifty 500 universe.")
    return constituents


def load_nifty500_constituents() -> tuple[pd.DataFrame, str]:
    """Return the current Nifty 500 list, with a bundled 500-stock fallback.

    The official CSV is tried first.  The packaged snapshot lets the stock
    selector remain complete when the NSE index endpoint is unavailable.
    """

    request_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NSE-Strategy-Lab/1.0)",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.niftyindices.com/",
    }
    for url in OFFICIAL_NIFTY500_URLS:
        try:
            request = Request(url, headers=request_headers)
            with urlopen(request, timeout=12) as response:
                constituents = _parse_constituents_csv(response.read())
            return constituents, "Official NSE Indices constituent file (live)"
        except Exception:
            continue

    if not PACKAGED_NIFTY500_CSV.exists():
        raise ValueError("The official Nifty 500 list could not be reached and no packaged list is available.")
    return _parse_constituents_csv(PACKAGED_NIFTY500_CSV.read_bytes()), "Packaged Nifty 500 constituent snapshot"


def _parse_bse500_payload(raw_payload: bytes) -> pd.DataFrame:
    """Validate and normalize BSE 500 constituent JSON returned by BSE Indices."""

    payload = json.loads(raw_payload.decode("utf-8"))
    rows = payload.get("Table", [])
    constituents = pd.DataFrame(rows)
    required = {"SCRIP_CODE", "SCRIPNAME"}
    missing = required.difference(constituents.columns)
    if missing:
        raise ValueError(f"BSE constituent response is missing: {', '.join(sorted(missing))}.")
    constituents = constituents.rename(
        columns={"SCRIP_CODE": "Scrip Code", "SCRIPNAME": "Company Name", "Industry_name": "Industry"}
    )
    keep = [column for column in ("Company Name", "Industry", "Scrip Code") if column in constituents]
    constituents = constituents.loc[:, keep].copy()
    constituents["Scrip Code"] = constituents["Scrip Code"].astype(str).str.strip().str.zfill(6)
    constituents["Company Name"] = constituents["Company Name"].astype(str).str.strip()
    constituents = constituents.drop_duplicates(subset="Scrip Code").sort_values("Company Name").reset_index(drop=True)
    if len(constituents) < 490:
        raise ValueError(f"Only {len(constituents)} BSE constituents were found; this is not a complete BSE 500 universe.")
    return constituents


def load_bse500_constituents() -> tuple[pd.DataFrame, str]:
    """Return the BSE 500 universe, with a bundled complete fallback snapshot."""

    request = Request(
        BSE500_CONSTITUENTS_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; NSE-Strategy-Lab/1.0)", "Referer": "https://www.bseindices.com/"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            return _parse_bse500_payload(response.read()), "Official BSE Indices constituent file (live)"
    except Exception:
        if not PACKAGED_BSE500_JSON.exists():
            raise ValueError("The BSE 500 list could not be reached and no packaged list is available.")
        return _parse_bse500_payload(PACKAGED_BSE500_JSON.read_bytes()), "Packaged BSE 500 constituent snapshot"


def _crosses_above(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left > right) & (left.shift(1) <= right.shift(1))


def _crosses_below(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left < right) & (left.shift(1) >= right.shift(1))


def _events_to_signal(buy: pd.Series, sell: pd.Series) -> pd.Series:
    """Return +1 BUY, -1 SELL, or 0 HOLD; conflict days remain HOLD."""

    signal = pd.Series(0, index=buy.index, dtype="int64")
    signal.loc[buy & ~sell] = 1
    signal.loc[sell & ~buy] = -1
    return signal


def calculate_indicators(prices: pd.DataFrame, params: StrategyParameters) -> pd.DataFrame:
    """Add indicator values plus persistent signal states and transition events."""

    params.validate()
    frame = prices.copy().sort_index()
    close = frame["close"].astype(float)

    frame["fast_ma"] = close.rolling(params.fast_window, min_periods=params.fast_window).mean()
    frame["slow_ma"] = close.rolling(params.slow_window, min_periods=params.slow_window).mean()
    frame["ma_event"] = _events_to_signal(
        _crosses_above(frame["fast_ma"], frame["slow_ma"]),
        _crosses_below(frame["fast_ma"], frame["slow_ma"]),
    )
    frame["ma_signal"] = 0
    frame.loc[frame["fast_ma"] > frame["slow_ma"], "ma_signal"] = 1
    frame.loc[frame["fast_ma"] < frame["slow_ma"], "ma_signal"] = -1

    change = close.diff()
    gains = change.clip(lower=0.0)
    losses = -change.clip(upper=0.0)
    # Wilder-style exponentially weighted rolling averages, commonly used for RSI.
    avg_gain = gains.ewm(alpha=1 / params.rsi_window, adjust=False, min_periods=params.rsi_window).mean()
    avg_loss = losses.ewm(alpha=1 / params.rsi_window, adjust=False, min_periods=params.rsi_window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    frame["rsi"] = 100 - (100 / (1 + rs))
    frame.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100.0
    frame.loc[(avg_gain == 0) & (avg_loss > 0), "rsi"] = 0.0
    frame.loc[(avg_gain == 0) & (avg_loss == 0), "rsi"] = 50.0
    frame["rsi_event"] = _events_to_signal(
        _crosses_below(frame["rsi"], pd.Series(params.rsi_lower, index=frame.index)),
        _crosses_above(frame["rsi"], pd.Series(params.rsi_upper, index=frame.index)),
    )
    frame["rsi_signal"] = 0
    frame.loc[frame["rsi"] <= params.rsi_lower, "rsi_signal"] = 1
    frame.loc[frame["rsi"] >= params.rsi_upper, "rsi_signal"] = -1

    frame["bb_middle"] = close.rolling(params.bollinger_window, min_periods=params.bollinger_window).mean()
    bb_sigma = close.rolling(params.bollinger_window, min_periods=params.bollinger_window).std(ddof=0)
    frame["bb_upper"] = frame["bb_middle"] + params.bollinger_std * bb_sigma
    frame["bb_lower"] = frame["bb_middle"] - params.bollinger_std * bb_sigma
    frame["bb_event"] = _events_to_signal(
        _crosses_below(close, frame["bb_lower"]),
        _crosses_above(close, frame["bb_upper"]),
    )
    frame["bb_signal"] = 0
    frame.loc[close <= frame["bb_lower"], "bb_signal"] = 1
    frame.loc[close >= frame["bb_upper"], "bb_signal"] = -1
    return frame


STRATEGIES: dict[str, tuple[str, ...]] = {
    "MA only": ("ma_signal",),
    "RSI only": ("rsi_signal",),
    "Bollinger only": ("bb_signal",),
    "MA + RSI": ("ma_signal", "rsi_signal"),
    "MA + Bollinger": ("ma_signal", "bb_signal"),
    "RSI + Bollinger": ("rsi_signal", "bb_signal"),
    "MA + RSI + Bollinger": ("ma_signal", "rsi_signal", "bb_signal"),
}


def aggregate_signals(frame: pd.DataFrame, indicator_columns: Iterable[str]) -> pd.DataFrame:
    """Apply strict agreement to persistent indicator states.

    Every selected module must be in BUY state to enter and every one must be
    in SELL state to exit.  A final signal is emitted only when consensus first
    changes, avoiding repeated BUY/SELL markers on every matching day.
    """

    columns = tuple(indicator_columns)
    if not columns:
        raise ValueError("At least one indicator must be selected.")
    result = frame.copy()
    signals = result.loc[:, columns]
    result["consensus_state"] = 0
    result.loc[(signals == 1).all(axis=1), "consensus_state"] = 1
    result.loc[(signals == -1).all(axis=1), "consensus_state"] = -1
    previous_state = result["consensus_state"].shift(1)
    result["final_signal"] = 0
    result.loc[(result["consensus_state"] == 1) & (previous_state != 1), "final_signal"] = 1
    result.loc[(result["consensus_state"] == -1) & (previous_state != -1), "final_signal"] = -1
    result["decision"] = result["final_signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
    result["consensus"] = result["consensus_state"].map({1: "BUY regime", -1: "SELL regime", 0: "NO AGREEMENT"})
    return result


def _empty_metrics() -> dict[str, float | int]:
    return {
        "cumulative_return": 0.0,
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "number_of_trades": 0,
        "buy_hold_return": 0.0,
        "buy_hold_annualized_return": 0.0,
    }


def _closed_trade_returns(close: pd.Series, position: pd.Series) -> list[float]:
    """Calculate realised long trade returns from position transitions."""

    entries = position.diff().fillna(position).eq(1)
    exits = position.diff().fillna(0).eq(-1)
    entry_price: float | None = None
    results: list[float] = []
    for timestamp, price in close.items():
        if entries.loc[timestamp] and entry_price is None:
            entry_price = float(price)
        if exits.loc[timestamp] and entry_price is not None:
            results.append(float(price) / entry_price - 1)
            entry_price = None
    return results


def run_backtest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Backtest a long/cash strategy, acting on the next session after a signal."""

    if "final_signal" not in frame:
        raise ValueError("Run aggregate_signals before running a backtest.")
    result = frame.copy()
    target_position = pd.Series(np.nan, index=result.index, dtype=float)
    target_position.loc[result["final_signal"] == 1] = 1.0
    target_position.loc[result["final_signal"] == -1] = 0.0
    result["position"] = target_position.ffill().fillna(0.0)

    result["asset_return"] = result["close"].pct_change().fillna(0.0)
    # Position is lagged one session: signal at today's close affects tomorrow.
    result["strategy_return"] = result["asset_return"] * result["position"].shift(1).fillna(0.0)
    result["strategy_equity"] = (1 + result["strategy_return"]).cumprod()
    result["buy_hold_equity"] = (1 + result["asset_return"]).cumprod()
    result["drawdown"] = result["strategy_equity"] / result["strategy_equity"].cummax() - 1

    periods = len(result)
    if periods < 2:
        return result, _empty_metrics()
    strategy_total = float(result["strategy_equity"].iloc[-1] - 1)
    buy_hold_total = float(result["buy_hold_equity"].iloc[-1] - 1)
    annualized = float(result["strategy_equity"].iloc[-1] ** (TRADING_DAYS / periods) - 1)
    buy_hold_annualized = float(result["buy_hold_equity"].iloc[-1] ** (TRADING_DAYS / periods) - 1)
    daily_volatility = float(result["strategy_return"].std(ddof=0))
    volatility = float(daily_volatility * np.sqrt(TRADING_DAYS))
    sharpe = (
        float(result["strategy_return"].mean() / daily_volatility * np.sqrt(TRADING_DAYS))
        if daily_volatility > 0 and np.isfinite(daily_volatility)
        else 0.0
    )
    trade_returns = _closed_trade_returns(result["close"], result["position"])
    win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0
    metrics = {
        "cumulative_return": strategy_total,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(result["drawdown"].min()),
        "win_rate": win_rate,
        "number_of_trades": len(trade_returns),
        "buy_hold_return": buy_hold_total,
        "buy_hold_annualized_return": buy_hold_annualized,
    }
    return result, metrics


def evaluate_strategy(prices: pd.DataFrame, params: StrategyParameters, strategy_name: str) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Calculate indicators, aggregate a named combination, and backtest it."""

    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    indicators = calculate_indicators(prices, params)
    final_signals = aggregate_signals(indicators, STRATEGIES[strategy_name])
    return run_backtest(final_signals)


def metrics_for_display(metrics: dict[str, float | int]) -> dict[str, str | int]:
    """Format metrics for a table without mutating the numerical source values."""

    return {
        "Cumulative return": f"{metrics['cumulative_return']:.1%}",
        "Annualized return": f"{metrics['annualized_return']:.1%}",
        "Annualized volatility": f"{metrics['annualized_volatility']:.1%}",
        "Sharpe ratio": f"{metrics['sharpe_ratio']:.2f}",
        "Max drawdown": f"{metrics['max_drawdown']:.1%}",
        "Win rate": f"{metrics['win_rate']:.1%}",
        "Completed trades": int(metrics["number_of_trades"]),
        "Buy & hold return": f"{metrics['buy_hold_return']:.1%}",
    }
