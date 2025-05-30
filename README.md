# Gap Guardian Strategy Backtester

This Streamlit application backtests the "Gap Guardian" trading strategy on specified financial symbols using historical data from Yahoo Finance.

## Strategy Overview

* **Symbols**: User-selectable (e.g., XAUUSD via "GC=F", S&P 500 via "^GSPC", NASDAQ via "^IXIC").
* **Account**: Customizable initial capital (default: $100,000).
* **Risk per Trade**: Customizable percentage (default: 0.5%).
* **Stop Loss Distance**: Customizable in price points (e.g., 15 points).
* **Trade Frequency**: Maximum one trade per day.
* **Time Frame**: 15 minutes.
* **Entry Window**: 9:30 AM to 11:00 AM New York time.
* **Entry Logic**:
    * Identifies the high and low of the 9:30 AM (NY time) 15-minute bar (opening range).
    * **Long Entry**: If price breaks below the opening range low and then closes back above it within the 9:30-11:00 AM window.
    * **Short Entry**: If price breaks above the opening range high and then closes back below it within the 9:30-11:00 AM window.
* **Exit Logic**:
    * Fixed 1:3 Risk/Reward Ratio.
    * Stop loss hit.

## Features

* Interactive UI for setting backtesting parameters.
* Dynamic data loading from Yahoo Finance.
* Detailed trade log.
* Equity curve visualization.
* Performance metrics (Total P&L, Win Rate, etc.).
* Light and Dark theme support.
* Modular code structure for easy maintenance and extension.

## Project Structure

gap_guardian_backtester/├── .streamlit/│   └── config.toml         # Streamlit theme and app configuration├── app.py                  # Main Streamlit application UI and flow├── config/│   ├── init.py│   └── settings.py         # Application constants and default parameters├── services/│   ├── init.py│   ├── data_loader.py      # Handles fetching and preparing market data│   ├── strategy_engine.py  # Implements the core trading strategy logic│   └── backtester.py       # Executes the backtest, simulates trades, calculates P&L├── utils/│   ├── init.py│   ├── logger.py           # Logging configuration│   └── plotting.py         # Functions for generating visualizations (equity curve, trades)├── static/│   └── style.css           # Custom CSS for styling the application├── .env.example            # Example environment variables (if any needed in future)├── .gitignore              # Specifies intentionally untracked files that Git should ignore├── README.md               # This file: project documentation└── requirements.txt        # Python package dependencies
## Setup and Installation

1.  **Clone the repository (Optional if deploying directly from GitHub to Streamlit Cloud):**
    ```bash
    git clone <repository_url>
    cd gap_guardian_backtester
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Usage

1.  Open the application in your browser (typically `http://localhost:8501`).
2.  Configure the backtesting parameters in the sidebar:
    * Select a symbol.
    * Set the start and end dates for the backtest (Note: `yfinance` provides up to 60 days of 15-minute intraday data).
    * Adjust initial capital, risk per trade, and stop loss points.
3.  Click the "Run Backtest" button.
4.  Review the results, including performance metrics, trade log, and equity curve.

## Customization and Theming

* **Themes**: The application supports light and dark themes. Configure these in `.streamlit/config.toml`.
* **Styling**: Custom CSS is located in `static/style.css`. Modify this file to change the application's appearance.

## Future Enhancements

* Support for more asset classes and exchanges.
* Advanced performance metrics (Sharpe Ratio, Sortino Ratio, Max Drawdown).
* Optimization module for strategy parameters.
* Integration with a persistent database for storing results.
* More sophisticated "false move" detection algorithms.
* Walk-forward analysis.
* Error tracking and more robust logging.
* Unit and integration tests.
