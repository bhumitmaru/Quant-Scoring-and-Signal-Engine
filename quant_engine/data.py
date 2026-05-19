"""Load single-ticker OHLCV from CSV with automatic column detection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

import pandas as pd

OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_column(columns: list[str], patterns: list[str], exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    for col in columns:
        if col in exclude:
            continue
        norm = _norm_col(col)
        for pat in patterns:
            if norm == pat:
                return col
    for col in columns:
        if col in exclude:
            continue
        norm = _norm_col(col)
        for pat in patterns:
            if len(pat) >= 4 and norm.startswith(pat):
                return col
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = list(df.columns)
    used: set[str] = set()

    dt_col = _find_column(cols, ["datetime", "dateandtime", "timestamp", "date", "time"])
    if dt_col:
        used.add(dt_col)
    unix_col = _find_column(cols, ["unixtimestamp", "unix", "epoch"], exclude=used)

    return {
        "datetime": dt_col,
        "unix": unix_col,
        "open": _find_column(cols, ["open", "o"], exclude=used),
        "high": _find_column(cols, ["high", "h"], exclude=used),
        "low": _find_column(cols, ["low", "l"], exclude=used),
        "close": _find_column(cols, ["close", "adjclose", "c", "price", "ltp"], exclude=used),
        "volume": _find_column(cols, ["volumeusd", "volume"])
        or _find_column(cols, ["vol", "v"], exclude=used)
        or _find_column(cols, ["volumebtc", "amount", "qty"], exclude=used),
    }


def infer_freq(index: pd.DatetimeIndex) -> str:
    if len(index) < 2:
        return "1d"
    delta = pd.Series(index).diff().median()
    if pd.isna(delta):
        return "1d"
    seconds = delta.total_seconds()
    if seconds <= 3600 * 2:
        return "1h"
    if seconds <= 86400 * 2:
        return "1d"
    if seconds <= 86400 * 8:
        return "1w"
    return "1me"


def periods_per_year(freq: str) -> int:
    f = freq.lower()
    if f.endswith("h"):
        return 24 * 365
    if f.endswith("d"):
        return 365
    if f.endswith("w"):
        return 52
    return 12


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows and leading flat/zero-volume history."""
    out = df.copy()
    out = out.replace([float("inf"), float("-inf")], float("nan"))
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[(out["Open"] > 0) & (out["High"] > 0) & (out["Low"] > 0) & (out["Close"] > 0)]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    if "Volume" in out.columns and out["Volume"].sum() > 0:
        first_active = out.index[out["Volume"] > 0]
        if len(first_active):
            out = out.loc[first_active[0] :]
    return out


def trim_backtest_window(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Drop leading bars where price is unrealistically low vs today's price
    (fixes glitched returns on assets like BTC that started at cents).
    """
    if df.empty:
        return df, None
    final = float(df["Close"].iloc[-1])
    threshold = max(final * 0.01, 1.0)
    if float(df["Close"].iloc[0]) >= threshold:
        return df, None
    valid = df[df["Close"] >= threshold]
    if valid.empty:
        return df, None
    start = valid.index[0]
    return df.loc[start:], str(start)


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday bars to daily OHLCV."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return df
    if infer_freq(df.index) == "1d":
        return df

    daily = df.resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return daily.dropna(subset=["Close"])


def load_ohlcv_csv(
    source: BinaryIO | str | Path,
    ticker_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    if isinstance(source, (str, Path)):
        raw = pd.read_csv(source)
        source_name = Path(source).stem
    else:
        raw = pd.read_csv(source)
        source_name = getattr(source, "name", "TICKER")
        if source_name and "." in str(source_name):
            source_name = Path(str(source_name)).stem

    detected = detect_columns(raw)
    missing = [k for k in ("open", "high", "low", "close") if not detected[k]]
    if missing:
        raise ValueError(
            f"Could not detect required columns: {', '.join(missing)}. "
            f"Found columns: {list(raw.columns)}"
        )

    df = pd.DataFrame()
    df["Open"] = pd.to_numeric(raw[detected["open"]], errors="coerce")
    df["High"] = pd.to_numeric(raw[detected["high"]], errors="coerce")
    df["Low"] = pd.to_numeric(raw[detected["low"]], errors="coerce")
    df["Close"] = pd.to_numeric(raw[detected["close"]], errors="coerce")
    df["Volume"] = (
        pd.to_numeric(raw[detected["volume"]], errors="coerce")
        if detected["volume"]
        else 0.0
    )

    if detected["datetime"]:
        df.index = pd.to_datetime(raw[detected["datetime"]], errors="coerce")
    elif detected["unix"]:
        ts = pd.to_numeric(raw[detected["unix"]], errors="coerce")
        df.index = pd.to_datetime(ts, unit="s", errors="coerce")
    else:
        df.index = pd.RangeIndex(len(df))

    df = clean_ohlcv(df)
    rows_before_resample = len(df)
    raw_freq = infer_freq(df.index) if isinstance(df.index, pd.DatetimeIndex) else "1d"
    resampled = raw_freq != "1d"
    if resampled:
        df = resample_to_daily(df)

    df, trim_start = trim_backtest_window(df)

    freq = infer_freq(df.index) if isinstance(df.index, pd.DatetimeIndex) else "1d"
    ticker = (ticker_name or source_name or "TICKER").upper()

    meta = {
        "ticker": ticker,
        "rows": len(df),
        "raw_rows": rows_before_resample,
        "freq": freq,
        "raw_freq": raw_freq,
        "resampled": resampled,
        "trimmed_from": trim_start,
        "detected": detected,
        "start": str(df.index[0]) if len(df) else None,
        "end": str(df.index[-1]) if len(df) else None,
    }
    return df, meta
