# Quant Strategy Backtest Engine

A Python backtesting app that runs a multi-factor technical strategy on **single-ticker OHLCV CSV** data. Upload your file, configure capital and fees, and view performance metrics, an equity curve vs buy-and-hold, and a full trade log.

Built with **vectorbt**, **TA-Lib**, **NumPy**, **Pandas**, and a **Streamlit** web UI.

---

## Features

- **Automatic column detection** — supports common OHLCV naming (`OPEN`, `HIGH`, `LOW`, `CLOSE`, `DATETIME`, `VOLUME_USD`, etc.)
- **Single-ticker backtest** — one CSV, full history
- **Configurable settings** — initial capital and commission (%)
- **Multi-factor strategy** — EMA trend, RSI, Bollinger Bands, MACD, pullback logic
- **Smart data handling**
  - Hourly (or intraday) data resampled to **daily** bars for stable indicators
  - Early low-price history trimmed (e.g. penny-era BTC) to avoid distorted returns
- **Clear results**
  - Final capital, total return, buy & hold benchmark
  - Sharpe, Sortino, max drawdown, win rate, profit factor
  - Equity curve indexed to 100 (strategy vs buy & hold)
  - Downloadable results CSV and trade log

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| [Streamlit](https://streamlit.io/) | Web interface |
| [vectorbt](https://vectorbt.dev/) | Vectorized backtesting |
| [TA-Lib](https://ta-lib.org/) | Technical indicators (EMA, RSI, BB, MACD, ATR) |
| [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) | Data pipeline |
| [Numba](https://numba.pydata.org/) | Fast numerical helpers (scoring module) |
| [QuantStats](https://github.com/ranaroussi/quantstats) | Return analytics |
| [PyFolio](https://github.com/stefan-jansen/pyfolio-reloaded) | Risk metrics (optional) |

---

## Requirements

- Python 3.9+
- macOS / Linux / Windows

**TA-Lib** needs the C library installed before `pip install TA-Lib`:

```bash
# macOS (Homebrew)
brew install ta-lib

# Ubuntu / Debian
sudo apt-get install ta-lib
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/quant-strategy-backtest.git
cd quant-strategy-backtest

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

1. Start the app:

```bash
streamlit run app.py
```

2. Open the URL shown in the terminal (usually `http://localhost:8501`).

3. In the sidebar, set:
   - **Initial funds** — starting capital (default: 100,000)
   - **Fees (%)** — commission per trade (e.g. `0.1` = 0.1%)
   - **Ticker name** (optional) — label for reports; defaults to the CSV filename

4. Upload your **OHLCV CSV** (single ticker).

5. Review backtest results, equity chart, and trade log. Export the summary CSV if needed.

---

## CSV Format

The engine auto-detects columns. Your file must include **Open, High, Low, Close** (any common spelling).

### Example: hourly Bitcoin data

```csv
UNIX_TIMESTAMP,DATETIME,OPEN,HIGH,CLOSE,LOW,VOLUME_USD,VOLUME_BTC
1279411200,2010-07-18 00:00:00,0.04951,0.04951,0.04951,0.04951,0.0,0.0
...
```

### Supported column names (case-insensitive)

| Field | Recognized names |
|-------|------------------|
| Open | `open`, `o` |
| High | `high`, `h` |
| Low | `low`, `l` |
| Close | `close`, `c`, `price`, `ltp` |
| Volume | `volume`, `volume_usd`, `volume_btc`, `vol` |
| Datetime | `datetime`, `date`, `timestamp`, `time` |
| Unix time | `unix_timestamp`, `unix`, `epoch` |

---

## Strategy Overview

The strategy combines five weighted factors into a **score (0–5)**:

1. **Trend** — price vs EMA 20 / EMA 50  
2. **Momentum** — RSI zones  
3. **Mean reversion** — Bollinger %B  
4. **Volatility** — distance to EMA 20 (pullback zone)  
5. **MACD** — histogram direction  

**Entry:** score ≥ 3.5 and (EMA pullback or RSI &lt; 40)  
**Exit:** score ≤ 1.8 or (price below EMA 20 and RSI &gt; 65)

Backtests use **100% of capital per position** with fees applied on each trade.

---

## Project Structure

```
.
├── app.py                 # Streamlit UI
├── requirements.txt
├── sample_chartink.csv    # Sample ticker list (optional)
└── quant_engine/
    ├── data.py            # CSV load, column detection, resampling
    ├── indicators.py      # TA-Lib indicators
    ├── strategy.py        # Entry/exit signal logic
    ├── backtest.py        # vectorbt backtest runner
    └── signals.py         # Legacy signal helpers
```

---

## Results Explained

| Metric | Description |
|--------|-------------|
| **Final capital** | Portfolio value at the end of the backtest |
| **Total return %** | Strategy return over the test period |
| **Buy & hold %** | Return from holding the asset with no trades |
| **Sharpe / Sortino** | Risk-adjusted return |
| **Max drawdown %** | Largest peak-to-trough decline |
| **Win rate %** | Percentage of profitable trades |
| **Equity curve** | Strategy vs buy & hold, both starting at 100 |

---

## Command-Line Backtest (optional)

```bash
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from quant_engine.data import load_ohlcv_csv
from quant_engine.backtest import run_single_backtest

ohlcv, meta = load_ohlcv_csv('path/to/your-data.csv')
result = run_single_backtest(ohlcv, meta['ticker'], init_cash=100_000, fees_pct=0.1)
print(result['stats'])
"
```

---

## Disclaimer

This project is for **research and education only**. Past performance does not guarantee future results. Not financial advice. Always validate strategies and data before live trading.

---

## License

MIT License — see [LICENSE](LICENSE) if included, or add your preferred license.

---

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/my-change`)  
3. Commit your changes (`git commit -m 'Add my change'`)  
4. Push to the branch (`git push origin feature/my-change`)  
5. Open a Pull Request  

---

## Author

**Himang Proj** — Quant scoring & signal backtest engine.
