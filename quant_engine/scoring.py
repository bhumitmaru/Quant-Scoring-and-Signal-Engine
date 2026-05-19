"""Multi-factor quant scoring with Numba acceleration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from quant_engine.data import fetch_universe
from quant_engine.indicators import add_indicators
from quant_engine.signals import build_vectorbt_signals


@njit(cache=True)
def _compute_factor_scores(
    close: float,
    ema20: float,
    ema50: float,
    rsi: float,
    bb_upper: float,
    bb_lower: float,
    macd_hist: float,
) -> tuple[float, float, float, float, float]:
    """Return sub-scores in [0, 1] for trend, momentum, mean-reversion, volatility, MACD."""
    trend = 0.0
    if ema20 > 0 and ema50 > 0:
        if close > ema20 > ema50:
            trend = 1.0
        elif close > ema20:
            trend = 0.65
        elif close < ema20 < ema50:
            trend = 0.0
        else:
            trend = 0.35

    momentum = 0.0
    if 45.0 <= rsi <= 60.0:
        momentum = 1.0
    elif 35.0 <= rsi < 45.0 or 60.0 < rsi <= 70.0:
        momentum = 0.6
    elif rsi < 30.0:
        momentum = 0.75
    elif rsi > 70.0:
        momentum = 0.1
    else:
        momentum = 0.4

    bb_width = bb_upper - bb_lower
    mean_rev = 0.5
    if bb_width > 0:
        pct_b = (close - bb_lower) / bb_width
        if 0.2 <= pct_b <= 0.45:
            mean_rev = 1.0
        elif pct_b < 0.15:
            mean_rev = 0.85
        elif pct_b > 0.85:
            mean_rev = 0.15
        else:
            mean_rev = 0.5

    ema_dist = abs(close - ema20) / ema20 if ema20 > 0 else 1.0
    volatility = 1.0 if ema_dist < 0.01 else (0.7 if ema_dist < 0.02 else 0.4)

    macd_score = 1.0 if macd_hist > 0 else (0.3 if macd_hist < 0 else 0.5)

    return trend, momentum, mean_rev, volatility, macd_score


@njit(cache=True)
def _aggregate_score(
    trend: float,
    momentum: float,
    mean_rev: float,
    volatility: float,
    macd_score: float,
) -> float:
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    factors = np.array([trend, momentum, mean_rev, volatility, macd_score])
    return float(np.dot(weights, factors) * 5.0)


@njit(cache=True)
def _signal_from_score(score: float, close: float, ema20: float, rsi: float) -> int:
    """
    Map score to signal code: 1=BUY, -1=SELL, 0=HOLD.
    Pullback-to-EMA buy zone from the PDF spec.
    """
    ema_pullback = (
        close > ema20 and ema20 > 0 and (close - ema20) / ema20 < 0.01
    )
    if score >= 3.5 and (ema_pullback or rsi < 40.0):
        return 1
    if score <= 1.8 or (close < ema20 and rsi > 65.0):
        return -1
    return 0


_SIGNAL_MAP = {1: "BUY", -1: "SELL", 0: "HOLD"}


def run_quant_analysis(df: pd.DataFrame) -> tuple[str, float, dict[str, Any]]:
    """
    Run multi-factor scoring on a single ticker's OHLCV history.
    Returns (signal, quant_score, factor_breakdown).
    """
    enriched = add_indicators(df)
    if enriched.empty:
        return "HOLD", 0.0, {}

    row = enriched.iloc[-1]
    close = float(row["Close"])
    ema20 = float(row["EMA_20"])
    ema50 = float(row["EMA_50"])
    rsi = float(row["RSI_14"])
    bb_upper = float(row["BB_UPPER"])
    bb_lower = float(row["BB_LOWER"])
    macd_hist = float(row["MACD_HIST"]) if not np.isnan(row["MACD_HIST"]) else 0.0

    trend, momentum, mean_rev, volatility, macd_score = _compute_factor_scores(
        close, ema20, ema50, rsi, bb_upper, bb_lower, macd_hist
    )
    score = _aggregate_score(trend, momentum, mean_rev, volatility, macd_score)
    code = _signal_from_score(score, close, ema20, rsi)
    signal = _SIGNAL_MAP[code]

    breakdown = {
        "Trend": round(trend * 5, 2),
        "Momentum": round(momentum * 5, 2),
        "Mean Reversion": round(mean_rev * 5, 2),
        "Volatility": round(volatility * 5, 2),
        "MACD": round(macd_score * 5, 2),
        "RSI": round(rsi, 2),
        "EMA20": round(ema20, 2),
    }
    return signal, round(score, 2), breakdown


def analyze_universe(
    universe: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Score all tickers and attach vectorbt confirmation flags."""
    rows: list[dict[str, Any]] = []

    for ticker, df in universe.items():
        signal, score, breakdown = run_quant_analysis(df)
        vbt_confirm = build_vectorbt_signals(df)

        rows.append(
            {
                "Ticker": ticker,
                "Last Close": round(float(df["Close"].iloc[-1]), 2),
                "Signal": signal,
                "Quant Score": score,
                "VBT Confirm": vbt_confirm,
                **{k: v for k, v in breakdown.items() if k in ("RSI", "EMA20")},
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Quant Score", ascending=False).reset_index(drop=True)
