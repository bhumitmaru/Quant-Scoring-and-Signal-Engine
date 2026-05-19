"""Technical indicators via TA-Lib."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA, RSI, and Bollinger Bands on OHLCV data."""
    out = df.copy()
    close = out["Close"].values.astype(np.float64)
    high = out["High"].values.astype(np.float64)
    low = out["Low"].values.astype(np.float64)

    out["EMA_20"] = talib.EMA(close, timeperiod=20)
    out["EMA_50"] = talib.EMA(close, timeperiod=50)
    out["RSI_14"] = talib.RSI(close, timeperiod=14)

    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    out["BB_UPPER"] = upper
    out["BB_MID"] = middle
    out["BB_LOWER"] = lower

    out["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = talib.MACD(close)

    return out.dropna()
