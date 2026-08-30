import numpy as np
import pandas as pd

from engine import (
    STRATEGIES,
    StrategyParameters,
    _parse_constituents_csv,
    aggregate_signals,
    calculate_indicators,
    evaluate_strategy,
)


def sample_prices() -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=180)
    # A deterministic oscillating trend that provides enough history for all indicators.
    close = 100 + np.linspace(0, 20, len(index)) + np.sin(np.linspace(0, 20, len(index))) * 8
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000}, index=index)


def test_indicator_frame_contains_all_module_signals():
    frame = calculate_indicators(sample_prices(), StrategyParameters())
    assert {"fast_ma", "slow_ma", "rsi", "bb_upper", "bb_lower", "ma_signal", "rsi_signal", "bb_signal"}.issubset(frame.columns)
    assert set(frame["ma_signal"].unique()).issubset({-1, 0, 1})


def test_aggregator_requires_unanimous_agreement():
    frame = pd.DataFrame({"a": [1, -1, 1, 0], "b": [1, -1, 0, 1]})
    actual = aggregate_signals(frame, ("a", "b"))["final_signal"].tolist()
    assert actual == [1, -1, 0, 0]


def test_state_agreement_emits_a_single_trade_event_per_regime_change():
    frame = pd.DataFrame({"a": [1, 1, 1, 0, -1, -1], "b": [1, 1, 1, 0, -1, -1]})
    result = aggregate_signals(frame, ("a", "b"))
    assert result["final_signal"].tolist() == [1, 0, 0, 0, -1, 0]
    assert result["consensus"].tolist() == ["BUY regime", "BUY regime", "BUY regime", "NO AGREEMENT", "SELL regime", "SELL regime"]


def test_each_declared_strategy_runs_and_returns_metrics():
    for name in STRATEGIES:
        frame, metrics = evaluate_strategy(sample_prices(), StrategyParameters(), name)
        assert {"final_signal", "strategy_equity", "buy_hold_equity"}.issubset(frame.columns)
        assert {"annualized_return", "sharpe_ratio", "number_of_trades"}.issubset(metrics)


def test_nifty_constituent_parser_keeps_a_complete_unique_universe():
    entries = "\n".join(f"Company {number},Industry,STOCK{number},EQ,ISIN{number}" for number in range(500))
    raw_csv = f"Company Name,Industry,Symbol,Series,ISIN Code\n{entries}\n".encode()
    universe = _parse_constituents_csv(raw_csv)
    assert len(universe) == 500
    assert universe["Symbol"].is_unique
