"""Vectorized signal generation via vectorbt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from quant_engine.indicators import add_indicators


def build_vectorbt_signals(df: pd.DataFrame) -> str:
    """
    Use vectorbt to confirm trend/momentum alignment on the latest bar.
    Returns YES / NO / NEUTRAL.
    """
    enriched = add_indicators(df)
    if len(enriched) < 25:
        return "NEUTRAL"

    close = enriched["Close"]
    ema20 = enriched["EMA_20"]
    rsi = enriched["RSI_14"]

    entries = (close > ema20) & (close.shift(1) <= ema20.shift(1)) & (rsi < 55)
    exits = (close < ema20) | (rsi > 70)

    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=100_000,
        fees=0.001,
        freq="1D",
    )

    if pf.trades.count() == 0:
        latest_bull = bool(close.iloc[-1] > ema20.iloc[-1] and rsi.iloc[-1] < 60)
        return "YES" if latest_bull else "NEUTRAL"

    last_trade = pf.trades.records_readable
    if last_trade.empty:
        return "NEUTRAL"

    last = last_trade.iloc[-1]
    if last["Status"] == "Open":
        return "YES"
    if last["Direction"] == "Long" and last["Status"] == "Closed":
        return "NEUTRAL"
    return "NO"


def backtest_ticker(df: pd.DataFrame) -> dict[str, float]:
    """Quick vectorbt backtest stats for a single series."""
    enriched = add_indicators(df)
    if len(enriched) < 25:
        return {}

    close = enriched["Close"]
    ema20 = enriched["EMA_20"]
    rsi = enriched["RSI_14"]

    entries = (close > ema20) & (rsi > 40) & (rsi < 65)
    exits = (close < ema20) | (rsi > 72)

    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=100_000,
        fees=0.001,
        freq="1D",
    )

    ret = pf.total_return()
    sharpe = pf.sharpe_ratio()
    mdd = pf.max_drawdown()

    return {
        "Total Return %": round(float(ret) * 100, 2) if not np.isnan(ret) else 0.0,
        "Sharpe": round(float(sharpe), 2) if sharpe is not None and not np.isnan(sharpe) else 0.0,
        "Max Drawdown %": round(float(mdd) * 100, 2) if not np.isnan(mdd) else 0.0,
    }
