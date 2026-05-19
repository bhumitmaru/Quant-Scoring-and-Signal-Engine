"""Performance analytics via QuantStats and PyFolio."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import quantstats as qs
from pyfolio import timeseries as pf_ts

from quant_engine.signals import backtest_ticker


def _close_returns(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].astype(float)
    return close.pct_change().dropna()


def quantstats_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Key QuantStats metrics for a single ticker."""
    returns = _close_returns(df)
    if returns.empty or len(returns) < 20:
        return {}

    qs.extend_pandas()
    return {
        "CAGR %": round(float(qs.stats.cagr(returns)) * 100, 2),
        "Sharpe": round(float(qs.stats.sharpe(returns)), 2),
        "Sortino": round(float(qs.stats.sortino(returns)), 2),
        "Max DD %": round(float(qs.stats.max_drawdown(returns)) * 100, 2),
        "Volatility %": round(float(qs.stats.volatility(returns)) * 100, 2),
        "Win Rate %": round(float(qs.stats.win_rate(returns)) * 100, 2),
    }


def pyfolio_risk_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """PyFolio-derived risk metrics."""
    returns = _close_returns(df)
    if returns.empty or len(returns) < 20:
        return {}

    return {
        "Annual Vol": round(float(pf_ts.annual_volatility(returns)) * 100, 2),
        "Calmar": round(float(pf_ts.calmar_ratio(returns)), 2)
        if not np.isnan(pf_ts.calmar_ratio(returns))
        else 0.0,
        "Omega": round(float(pf_ts.omega_ratio(returns)), 2),
        "Tail Ratio": round(float(pf_ts.tail_ratio(returns)), 2),
    }


def build_analytics_row(ticker: str, df: pd.DataFrame) -> dict[str, Any]:
    """Merge vectorbt backtest, QuantStats, and PyFolio for one ticker."""
    row: dict[str, Any] = {"Ticker": ticker}
    row.update(backtest_ticker(df))
    row.update(quantstats_summary(df))
    row.update(pyfolio_risk_metrics(df))
    return row


def portfolio_snapshot(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analytics table across the scanned universe."""
    rows = [build_analytics_row(t, df) for t, df in universe.items()]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
