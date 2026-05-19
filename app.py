"""
Backtest strategy on a single-ticker OHLCV CSV (auto-detects columns).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from quant_engine.backtest import run_single_backtest
from quant_engine.data import load_ohlcv_csv

st.set_page_config(page_title="Strategy Backtest", layout="wide")

st.title("Strategy Backtest Results")
st.caption("Upload OHLCV CSV → auto-detect columns → backtest on full history")


def _fmt_money(val: float) -> str:
    return f"{val:,.2f}"


with st.sidebar:
    st.header("Settings")
    init_cash = st.number_input("Initial funds", min_value=1_000, value=100_000, step=1_000)
    fees_pct = st.number_input(
        "Fees (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.01,
        format="%.2f",
    )
    ticker_override = st.text_input("Ticker name (optional)", value="")

uploaded_file = st.file_uploader("OHLCV CSV (single ticker)", type=["csv"])

if uploaded_file is not None:
    try:
        name = ticker_override.strip() or None
        ohlcv, meta = load_ohlcv_csv(uploaded_file, ticker_name=name)

        st.subheader(f"Data: {meta['ticker']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bars (backtest)", f"{meta['rows']:,}")
        c2.metric("Bar size", meta["freq"] + (" (from hourly)" if meta.get("resampled") else ""))
        c3.metric("From", str(meta["start"])[:10])
        c4.metric("To", str(meta["end"])[:10])

        notes = []
        if meta.get("resampled"):
            notes.append(f"Hourly data ({meta['raw_rows']:,} bars) → daily candles")
        if meta.get("trimmed_from"):
            notes.append(f"Skipped early low-price history (backtest starts {str(meta['trimmed_from'])[:10]})")
        if notes:
            st.caption(" · ".join(notes))

        detected = meta["detected"]
        with st.expander("Detected columns"):
            st.table(
                pd.DataFrame(
                    [
                        {"Field": "Open", "CSV column": detected.get("open")},
                        {"Field": "High", "CSV column": detected.get("high")},
                        {"Field": "Low", "CSV column": detected.get("low")},
                        {"Field": "Close", "CSV column": detected.get("close")},
                        {"Field": "Volume", "CSV column": detected.get("volume") or "(none)"},
                        {
                            "Field": "Datetime",
                            "CSV column": detected.get("datetime") or detected.get("unix"),
                        },
                    ]
                )
            )

        with st.spinner("Running backtest..."):
            result = run_single_backtest(
                ohlcv,
                meta["ticker"],
                init_cash=float(init_cash),
                fees_pct=float(fees_pct),
                freq=meta["freq"],
            )

        if not result["stats"]:
            st.error("No trades generated on this data.")
            st.stop()

        s = result["stats"]
        st.subheader("Backtest results")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Final capital", _fmt_money(s["Final Capital"]))
        m2.metric("Total return", f"{s['Total Return %']:.2f}%")
        m3.metric("Buy & hold", f"{s['Benchmark Return %']:.2f}%")
        m4.metric("Sharpe", f"{s['Sharpe']:.2f}")
        m5.metric("Trades", s["Total Trades"])

        show_cols = [
            "Initial Capital",
            "Final Capital",
            "Total Return %",
            "Benchmark Return %",
            "Annualized Return %",
            "Sharpe",
            "Sortino",
            "Max Drawdown %",
            "Win Rate %",
            "Total Trades",
            "Profit Factor",
            "Avg Trade Return %",
            "Bars",
            "Frequency",
        ]
        stats_df = pd.DataFrame({k: s[k] for k in show_cols if k in s}, index=["Value"]).T
        st.dataframe(stats_df, use_container_width=True)

        st.download_button(
            "Download results CSV",
            data=pd.DataFrame([s]).to_csv(index=False).encode("utf-8"),
            file_name=f"backtest_{meta['ticker']}_{dt.date.today()}.csv",
            mime="text/csv",
        )

        if not result["equity_norm"].empty:
            st.subheader("Equity curve (indexed to 100 at start)")
            chart_df = pd.DataFrame(
                {
                    "Strategy": result["equity_norm"],
                    "Buy & Hold": result["benchmark_norm"],
                }
            )
            st.line_chart(chart_df, use_container_width=True)

        if not result["trades"].empty:
            st.subheader("Trade log")
            display_cols = [
                c
                for c in [
                    "Entry Timestamp",
                    "Exit Timestamp",
                    "Avg Entry Price",
                    "Avg Exit Price",
                    "Size",
                    "PnL",
                    "Return %",
                    "Direction",
                    "Status",
                ]
                if c in result["trades"].columns
            ]
            st.dataframe(result["trades"][display_cols], use_container_width=True, height=400)

    except Exception as exc:
        st.error(f"Could not load CSV: {exc}")
        st.info("CSV must include Open, High, Low, and Close columns.")
else:
    st.info("Upload an OHLCV CSV file to run the backtest.")
