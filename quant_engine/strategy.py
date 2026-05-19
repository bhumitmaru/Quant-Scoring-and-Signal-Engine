from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.indicators import add_indicators


def strategy_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Build entry/exit boolean series from the quant scoring rules.
    Returns (entries, exits, enriched_df with SCORE column).
    """
    data = add_indicators(df)
    if data.empty:
        empty = pd.Series(dtype=bool)
        return empty, empty, data

    close = data["Close"]
    ema20 = data["EMA_20"]
    ema50 = data["EMA_50"]
    rsi = data["RSI_14"]
    bb_upper = data["BB_UPPER"]
    bb_lower = data["BB_LOWER"]
    macd_hist = data["MACD_HIST"].fillna(0)

    trend = np.where(
        (close > ema20) & (ema20 > ema50),
        1.0,
        np.where(close > ema20, 0.65, np.where((close < ema20) & (ema20 < ema50), 0.0, 0.35)),
    )

    momentum = np.select(
        [
            (rsi >= 45) & (rsi <= 60),
            ((rsi >= 35) & (rsi < 45)) | ((rsi > 60) & (rsi <= 70)),
            rsi < 30,
            rsi > 70,
        ],
        [1.0, 0.6, 0.75, 0.1],
        default=0.4,
    )

    bb_width = bb_upper - bb_lower
    pct_b = np.where(bb_width > 0, (close - bb_lower) / bb_width, 0.5)
    mean_rev = np.select(
        [
            (pct_b >= 0.2) & (pct_b <= 0.45),
            pct_b < 0.15,
            pct_b > 0.85,
        ],
        [1.0, 0.85, 0.15],
        default=0.5,
    )

    ema_dist = np.where(ema20 > 0, np.abs(close - ema20) / ema20, 1.0)
    volatility = np.where(ema_dist < 0.01, 1.0, np.where(ema_dist < 0.02, 0.7, 0.4))

    macd_score = np.where(macd_hist > 0, 1.0, np.where(macd_hist < 0, 0.3, 0.5))

    score = (
        0.30 * trend + 0.25 * momentum + 0.20 * mean_rev + 0.15 * volatility + 0.10 * macd_score
    ) * 5.0
    data["SCORE"] = score

    ema_pullback = (close > ema20) & (ema20 > 0) & ((close - ema20) / ema20 < 0.01)
    entries = (score >= 3.5) & (ema_pullback | (rsi < 40))
    exits = (score <= 1.8) | ((close < ema20) & (rsi > 65))

    return pd.Series(entries, index=data.index), pd.Series(exits, index=data.index), data
