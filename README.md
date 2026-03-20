# AMT Quantitative Signal Engine

Order Flow and Auction Market Theory (AMT) signal engine for detecting volume imbalances, trap zones, and session breakouts.

## Architecture

* **`data/`**: Ingestion scripts for Binance and CME/Alpaca (Tick level / 1m resolution).
* **`core/`**: Foundry for the mathematical calculation of Volume Profiles, VAH, VAL, POC, and CVD (Cumulative Volume Delta).
* **`signals/`**: Real-time evaluation of heuristic rules (e.g. \"Look Above and Fail\", Exhaustion traps).
* **`alerts/`**: Connectors for Discord, Telegram, or Webhooks.
* **`backtest/`**: Historical replay tools for order flow validation.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
2. add your API keys in a `.env` file:
   ```
   ALPACA_API_KEY=KEY
   ALPACA_SECRET_KEY=KEY
   ```
3. Run the real-time signal engine:
   python main.py