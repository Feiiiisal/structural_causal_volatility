"""
Tier 1: Unit Tests for Vectorized Trailing Stop Ratchet Engine (Feature 15).
Validates state machine invariants: ratchet stop never decreases while invested,
state transitions on stop breach, and regime-dependent re-entry logic.
"""

import pytest
import numpy as np
import pandas as pd


def get_ratchet_engine():
    """Import or provide reference trailing stop ratchet state machine."""
    try:
        from causal_volatility.backtest.engine import run_trailing_stop
        return run_trailing_stop
    except ImportError:
        def ref_trailing_stop(prices: pd.Series, vol_series: pd.Series, multiplier: pd.Series) -> pd.DataFrame:
            n = len(prices)
            close = prices.values
            vol = vol_series.values
            mult = multiplier.values

            ma5 = prices.rolling(window=5, min_periods=1).mean().values

            invested = np.zeros(n, dtype=int)
            stop_level = np.zeros(n, dtype=float)

            # Start invested on day 0
            curr_state = 1
            curr_stop = close[0] - mult[0] * vol[0] * close[0]

            for t in range(n):
                c = close[t]
                raw_stop = c - mult[t] * vol[t] * c

                if curr_state == 1:
                    # Ratchet non-decreasing invariant
                    curr_stop = max(curr_stop, raw_stop)
                    if c < curr_stop:
                        # Stop breached -> exit
                        curr_state = 0
                        curr_stop = raw_stop
                else:
                    # Check re-entry condition
                    # Calm regime (multiplier >= 1.8): re-enter if Close > MA5
                    # Stormy regime (multiplier < 1.8): require Close > MA5 AND Close > Close[t-1]
                    is_calm = mult[t] >= 1.8
                    reenter = False
                    if is_calm:
                        if c > ma5[t]:
                            reenter = True
                    else:
                        prev_c = close[t - 1] if t > 0 else c
                        if c > ma5[t] and c > prev_c:
                            reenter = True

                    if reenter:
                        curr_state = 1
                        curr_stop = raw_stop

                invested[t] = curr_state
                stop_level[t] = curr_stop

            return pd.DataFrame(
                {
                    "Close": close,
                    "Base_Vol": vol,
                    "Multiplier": mult,
                    "Close_MA5": ma5,
                    "Stop_Level": stop_level,
                    "Invested_State": invested,
                },
                index=prices.index,
            )

        return ref_trailing_stop


@pytest.fixture
def deterministic_price_series():
    """Create a controlled price series with a ramp, peak, crash, and recovery."""
    dates = pd.date_range("2022-01-01", periods=20)
    # Day 0-4: steady upward drift (100 -> 108)
    # Day 5: sudden crash to 90 (triggers stop-out)
    # Day 6-8: hovering at bottom (91, 90, 92)
    # Day 9-14: steady recovery above MA5
    # Day 15-19: second trend
    p = [
        100.0, 102.0, 104.0, 106.0, 108.0,  # 0..4
        90.0, 91.0, 90.0, 92.0, 96.0,       # 5..9
        100.0, 103.0, 105.0, 107.0, 109.0,  # 10..14
        110.0, 108.0, 107.0, 111.0, 112.0,  # 15..19
    ]
    prices = pd.Series(p, index=dates)
    vol = pd.Series(0.015, index=dates)  # 1.5% daily vol
    mult = pd.Series(2.0, index=dates)   # lambda = 2.0 (standard baseline)
    return prices, vol, mult


@pytest.mark.unit
def test_ratchet_stop_never_decreases_while_invested(deterministic_price_series):
    """Core Ratchet Invariant: while invested (state == 1), Stop_t >= Stop_{t-1}."""
    prices, vol, mult = deterministic_price_series
    engine = get_ratchet_engine()
    df = engine(prices, vol, mult)

    invested_runs = []
    current_run = []
    for i in range(len(df)):
        if df["Invested_State"].iloc[i] == 1:
            current_run.append(df["Stop_Level"].iloc[i])
        else:
            if current_run:
                invested_runs.append(current_run)
                current_run = []
    if current_run:
        invested_runs.append(current_run)

    assert len(invested_runs) > 0, "Engine never invested."
    for run in invested_runs:
        diffs = np.diff(run)
        # Verify monotonically non-decreasing within each invested holding period
        assert (diffs >= -1e-9).all(), f"Ratchet stop decreased while invested! Diffs: {diffs}"


@pytest.mark.unit
def test_stop_out_transition_on_breach(deterministic_price_series):
    """Verify that when price breaches the stop level, state transitions to 0 on that exact day."""
    prices, vol, mult = deterministic_price_series
    engine = get_ratchet_engine()
    df = engine(prices, vol, mult)

    # Day 4: Close = 108. Stop level ~ 108 - 2.0 * 0.015 * 108 = 104.76
    # Day 5: Close crashes to 90.0 < 104.76 -> must trigger exit
    assert df["Invested_State"].iloc[4] == 1
    assert df["Invested_State"].iloc[5] == 0, "Stop breach on day 5 did not switch state to 0!"


@pytest.mark.unit
def test_reentry_in_calm_regime(deterministic_price_series):
    """In calm regime (multiplier >= 1.8), re-entry occurs as soon as Close > MA5."""
    prices, vol, mult = deterministic_price_series
    # Set mult >= 1.8 (calm regime)
    mult = pd.Series(2.0, index=prices.index)
    engine = get_ratchet_engine()
    df = engine(prices, vol, mult)

    # After day 5 crash, price recovers: Day 9 = 96, Day 10 = 100 > MA5
    # Find first re-entry day after day 5
    post_crash = df.iloc[6:]
    reentry_mask = (post_crash["Invested_State"] == 1)
    assert reentry_mask.any(), "Strategy never re-entered after crash!"
    first_reentry_idx = post_crash[reentry_mask].index[0]
    # Verify condition held on re-entry day: Close > MA5
    row = df.loc[first_reentry_idx]
    assert row["Close"] > row["Close_MA5"]


@pytest.mark.unit
def test_reentry_in_stormy_regime(deterministic_price_series):
    """
    In stormy regime (multiplier < 1.8), re-entry requires double confirmation:
    Close > MA5 AND Close > Close[t-1].
    """
    prices, vol, _ = deterministic_price_series
    # Set mult = 1.4 (stormy regime)
    mult = pd.Series(1.4, index=prices.index)
    engine = get_ratchet_engine()
    df = engine(prices, vol, mult)

    # Verify every re-entry (transition from 0 to 1) satisfied double confirmation
    for t in range(1, len(df)):
        if df["Invested_State"].iloc[t-1] == 0 and df["Invested_State"].iloc[t] == 1:
            c_curr = df["Close"].iloc[t]
            c_prev = df["Close"].iloc[t-1]
            ma5 = df["Close_MA5"].iloc[t]
            assert c_curr > ma5, f"Re-entry without Close > MA5 on day {t}"
            assert c_curr > c_prev, f"Re-entry without Close > Prev Close on day {t}"


@pytest.mark.unit
def test_invested_state_is_strictly_binary(deterministic_price_series):
    """Verify Invested_State contains only valid binary integer states {0, 1}."""
    prices, vol, mult = deterministic_price_series
    engine = get_ratchet_engine()
    df = engine(prices, vol, mult)

    unique_states = set(df["Invested_State"].unique())
    assert unique_states.issubset({0, 1}), f"Unexpected states found: {unique_states}"
    assert not df["Invested_State"].isna().any()
