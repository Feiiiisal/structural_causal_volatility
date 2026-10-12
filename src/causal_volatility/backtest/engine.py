"""
Vectorized Trailing Stop Ratchet State Machine Engine (Feature 15).

Provides high-performance state machine simulation for adaptive risk bounds,
enforcing ratchet non-decreasing properties and regime-dependent re-entry logic.
"""

from typing import Optional
import numpy as np
import pandas as pd


def run_trailing_stop(
    prices: pd.Series,
    vol_series: pd.Series,
    multiplier: pd.Series,
    calm_threshold: float = 2.8,
    ma_window: int = 20,
    max_cash_days: Optional[int] = 15,
) -> pd.DataFrame:
    """
    Execute 1D vectorized trailing stop ratchet state machine.

    Supports both ATR-based stops (price-denominated) and percentage volatility stops.
    If vol_series mean < 0.2, it is treated as a percentage: Stop = Close * (1 - λ * vol).
    Otherwise, it is treated as ATR in price terms: Stop = Close - λ * ATR.

    Invariants:
    1. While invested (state == 1), Stop_t >= Stop_{t-1} (ratchet monotonicity).
    2. If Close_t < Stop_t, state immediately transitions to 0 (stop-out exit).
    3. Re-entry logic:
       - Calm regime (multiplier >= calm_threshold): Re-enters if Close_t > MA_t.
       - Stormy regime (multiplier < calm_threshold): Requires Close_t > MA_t AND Close_t > Close_{t-1}.
       - Optional forced re-entry after max_cash_days in cash.

    Args:
        prices: Asset close price series.
        vol_series: Volatility or ATR series.
        multiplier: Dynamic risk multiplier series lambda_t.
        calm_threshold: Multiplier threshold separating calm vs stormy regimes (default: 1.8).
        ma_window: Moving average window for re-entry confirmation (default: 5).
        max_cash_days: Maximum consecutive days in cash before forced re-entry (default: None).

    Returns:
        DataFrame indexed by date containing:
            - 'Close': close prices
            - 'Base_Vol': volatility/ATR input
            - 'Multiplier': multiplier lambda_t
            - 'Close_MA5': moving average of close (for backward compatibility)
            - 'Close_MA': moving average of close
            - 'Stop_Level': active ratchet stop level
            - 'Invested_State': binary position state (1 = invested, 0 = cash)
    """
    n = len(prices)
    close = prices.values.astype(float)
    vol_raw = vol_series.values.astype(float)
    mult = multiplier.values.astype(float)

    ma = prices.rolling(window=ma_window, min_periods=1).mean().values

    # Determine if vol is fractional (GK / return vol) or price-denominated (ATR)
    is_fractional = bool(np.nanmean(vol_raw) < 0.2)

    invested = np.zeros(n, dtype=int)
    stop_level = np.zeros(n, dtype=float)

    curr_state = 1
    v0 = vol_raw[0] * close[0] if is_fractional else vol_raw[0]
    curr_stop = close[0] - mult[0] * v0
    cash_days = 0

    for t in range(n):
        c = close[t]
        v_t = vol_raw[t] * c if is_fractional else vol_raw[t]
        raw_stop = c - mult[t] * v_t

        if curr_state == 1:
            # Ratchet non-decreasing invariant while invested
            curr_stop = max(curr_stop, raw_stop)
            cash_days = 0
            if c < curr_stop:
                # Stop breached -> transition to cash
                curr_state = 0
                curr_stop = raw_stop
                cash_days = 1
        else:
            # Re-entry evaluation from cash state
            cash_days += 1

            if max_cash_days is not None and cash_days >= max_cash_days:
                curr_state = 1
                curr_stop = raw_stop
                cash_days = 0
            else:
                is_calm = mult[t] >= calm_threshold
                reenter = False
                if is_calm:
                    if c > ma[t]:
                        reenter = True
                else:
                    prev_c = close[t - 1] if t > 0 else c
                    if c > ma[t] and c > prev_c:
                        reenter = True

                if reenter:
                    curr_state = 1
                    curr_stop = raw_stop
                    cash_days = 0

        invested[t] = curr_state
        stop_level[t] = curr_stop

    return pd.DataFrame(
        {
            "Close": close,
            "Base_Vol": vol_raw,
            "Multiplier": mult,
            "Close_MA5": ma,
            "Close_MA": ma,
            "Stop_Level": stop_level,
            "Invested_State": invested,
        },
        index=prices.index,
    )


def execute_causal_trailing_stop(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    lambda_t: pd.Series,
    static_multiplier: float = 4.5,
    calm_threshold: float = 2.8,
    ma_window: int = 20,
    max_cash_days: int = 15,
) -> pd.DataFrame:
    """
    Execute dual backtest comparing standard static baseline against causal dynamic stop.

    Args:
        df_raw: Raw DataFrame containing 'SP100_Close'.
        df_processed: Processed DataFrame containing 'ATR_14' or 'Garman_Klass_Vol'.
        lambda_t: Dynamic multiplier series from causal module.
        static_multiplier: Fixed multiplier for baseline (default: 4.5).
        calm_threshold: Calm regime threshold (default: 2.8).
        ma_window: Moving average window for re-entry (default: 20).
        max_cash_days: Max consecutive days in cash (default: 15).

    Returns:
        DataFrame indexed by date with both Standard and Causal stop levels and states.
    """
    common_idx = df_processed.index.intersection(lambda_t.index)
    prices = df_raw.loc[common_idx, "SP100_Close"]

    # Use ATR for stop band if available
    if "ATR_14" in df_processed.columns:
        vol_band = df_processed.loc[common_idx, "ATR_14"]
    else:
        vol_band = df_processed.loc[common_idx, "Garman_Klass_Vol"].rolling(5, min_periods=1).mean().bfill()

    mult_causal = lambda_t.loc[common_idx]
    mult_static = pd.Series(static_multiplier, index=common_idx)

    res_static = run_trailing_stop(
        prices, vol_band, mult_static,
        calm_threshold=calm_threshold, ma_window=ma_window, max_cash_days=max_cash_days,
    )
    res_causal = run_trailing_stop(
        prices, vol_band, mult_causal,
        calm_threshold=calm_threshold, ma_window=ma_window, max_cash_days=max_cash_days,
    )

    out_df = pd.DataFrame(
        {
            "Close": prices,
            "Base_Vol": vol_band,
            "Causal_Multiplier": mult_causal,
            "Close_MA5": res_causal["Close_MA5"],
            "Standard_Stop_Level": res_static["Stop_Level"],
            "Causal_Stop_Level": res_causal["Stop_Level"],
            "Standard_State": res_static["Invested_State"],
            "Causal_State": res_causal["Invested_State"],
        },
        index=common_idx,
    )
    return out_df
