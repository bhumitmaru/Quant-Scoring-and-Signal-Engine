"""Backtest the quant strategy on historical OHLCV via vectorbt."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from quant_engine.data import infer_freq
from quant_engine.strategy import strategy_signals


def _safe_float(val, default: float = 0.0, decimals: int = 2) -> float:
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return default
    return round(float(val), decimals)


def run_backtest(
    ticker: str,
    df: pd.DataFrame,
    init_cash: float = 100_000,
    fees_pct: float = 0.1,
    freq: str | None = None,
) -> dict[str, Any]:
    entries, exits, enriched = strategy_signals(df)
    if enriched.empty or not entries.any():
        return {
            "ticker": ticker,
            "stats": {},
            "trades": pd.DataFrame(),
            "equity": pd.Series(dtype=float),
            "equity_norm": pd.Series(dtype=float),
            "benchmark_norm": pd.Series(dtype=float),
            "returns": pd.Series(dtype=float),
        }

    close = enriched["Close"]
    bar_freq = freq or (
        infer_freq(df.index) if isinstance(df.index, pd.DatetimeIndex) else "1d"
    )
    fees = fees_pct / 100.0

    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        freq=bar_freq,
        size=1.0,
        size_type="percent",
        accumulate=False,
    )

    equity = pf.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]

    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    total_return_pct = (end_val / init_cash - 1.0) * 100.0

    bench_start = float(close.iloc[0])
    bench_end = float(close.iloc[-1])
    benchmark_return_pct = (bench_end / bench_start - 1.0) * 100.0

    equity_norm = (equity / start_val) * 100.0
    benchmark_norm = (close / bench_start) * 100.0

    sharpe = pf.sharpe_ratio()
    sortino = pf.sortino_ratio() if hasattr(pf, "sortino_ratio") else np.nan
    ann_return = pf.annualized_return()

    stats = {
        "Ticker": ticker,
        "Bars": len(close),
        "Frequency": bar_freq,
        "Initial Capital": _safe_float(init_cash),
        "Final Capital": _safe_float(end_val),
        "Total Return %": _safe_float(total_return_pct),
        "Benchmark Return %": _safe_float(benchmark_return_pct),
        "Annualized Return %": _safe_float(ann_return * 100),
        "Sharpe": _safe_float(sharpe, decimals=3),
        "Sortino": _safe_float(sortino, decimals=3),
        "Max Drawdown %": _safe_float(pf.max_drawdown() * 100),
        "Win Rate %": _safe_float(pf.trades.win_rate() * 100) if pf.trades.count() > 0 else 0.0,
        "Total Trades": int(pf.trades.count()),
        "Profit Factor": _safe_float(pf.trades.profit_factor()) if pf.trades.count() > 0 else 0.0,
        "Avg Trade Return %": _safe_float(pf.trades.returns.mean() * 100)
        if pf.trades.count() > 0
        else 0.0,
    }

    trades = pf.trades.records_readable.copy() if pf.trades.count() > 0 else pd.DataFrame()
    if not trades.empty and "Return" in trades.columns:
        trades["Return %"] = (trades["Return"] * 100).round(2)

    return {
        "ticker": ticker,
        "stats": stats,
        "trades": trades,
        "equity": equity,
        "equity_norm": equity_norm,
        "benchmark_norm": benchmark_norm,
        "returns": pf.returns().dropna(),
        "portfolio": pf,
    }


def run_single_backtest(
    df: pd.DataFrame,
    ticker: str,
    init_cash: float = 100_000,
    fees_pct: float = 0.1,
    freq: str | None = None,
) -> dict[str, Any]:
    return run_backtest(ticker, df, init_cash=init_cash, fees_pct=fees_pct, freq=freq)
