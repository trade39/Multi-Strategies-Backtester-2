# config/settings.py
"""
Application-wide constants and default parameters.
"""
import pytz
import numpy as np

# --- Strategy Settings ---
AVAILABLE_STRATEGIES = ["Gap Guardian", "Unicorn", "Silver Bullet"]
DEFAULT_STRATEGY = "Gap Guardian"

# Silver Bullet specific time windows (NY Time)
SILVER_BULLET_WINDOWS_NY = [
    (pytz.datetime.time(3, 0), pytz.datetime.time(4, 0)),
    (pytz.datetime.time(10, 0), pytz.datetime.time(11, 0)),
    (pytz.datetime.time(14, 0), pytz.datetime.time(15, 0)),
]
# Parameters for Unicorn strategy (e.g., lookback for swing points)
UNICORN_SWING_LOOKBACK = 5 # Number of bars on each side to define a swing point

# Default Tickers for yfinance
DEFAULT_TICKERS = {
    "Gold (XAU/USD)": "GC=F",
    "S&P 500 Index": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "EUR/USD": "EURUSD=X",
    "Bitcoin (BTC/USD)": "BTC-USD"
}

# Default Backtesting Parameters
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_RISK_PER_TRADE_PERCENT = 0.5
DEFAULT_STOP_LOSS_POINTS = 15.0 # General SL, can be adapted by strategy
DEFAULT_RRR = 3.0 # General RRR

# --- Timeframe Settings ---
AVAILABLE_TIMEFRAMES = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "Daily": "1d",
    "Weekly": "1wk",
}
DEFAULT_STRATEGY_TIMEFRAME = "15m"

# yfinance interval categories for history limits
YFINANCE_SHORT_INTRADAY_INTERVALS = ["1m", "2m", "5m", "15m", "30m"] # Typically up to 60 days
YFINANCE_HOURLY_INTERVALS = ["60m", "1h", "90m"] # "1h", "90m" often up to 730 days.

# Default Strategy Entry Window (NY Time) - Primarily for Gap Guardian
DEFAULT_ENTRY_WINDOW_START_HOUR = 9
DEFAULT_ENTRY_WINDOW_START_MINUTE = 30
DEFAULT_ENTRY_WINDOW_END_HOUR = 11
DEFAULT_ENTRY_WINDOW_END_MINUTE = 0

NY_TIMEZONE_STR = "America/New_York"
NY_TIMEZONE = pytz.timezone(NY_TIMEZONE_STR)

# Data Fetching Limits
MAX_SHORT_INTRADAY_DAYS = 60
MAX_HOURLY_INTRADAY_DAYS = 730

# Plotting
PLOTLY_TEMPLATE = "plotly_white" # Consider "plotly_dark" or theme-based selection

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO' # Change to 'DEBUG' for more verbose logs

# UI
APP_TITLE = "Multi-Strategy Backtester" # Updated App Title

# Metric Colors
POSITIVE_METRIC_COLOR = "#28a745"
NEGATIVE_METRIC_COLOR = "#dc3545"
NEUTRAL_METRIC_COLOR = "#FAFAFA" # For dark theme, consider a lighter gray like #adb5bd

# --- Optimization Settings ---
OPTIMIZATION_ALGORITHMS = ["Grid Search", "Random Search"]
DEFAULT_OPTIMIZATION_ALGORITHM = "Grid Search"
OPTIMIZATION_METRICS = ["Total P&L", "Profit Factor", "Win Rate", "Sharpe Ratio (Annualized)", "Sortino Ratio (Annualized)", "Max Drawdown (%)"]
DEFAULT_OPTIMIZATION_METRIC = "Sharpe Ratio (Annualized)"

# Parameter ranges for optimization (can be made strategy-specific later)
DEFAULT_SL_POINTS_OPTIMIZATION_RANGE = {"min": 5.0, "max": 50.0, "steps": 5} # Expanded range
DEFAULT_RRR_OPTIMIZATION_RANGE = {"min": 1.0, "max": 5.0, "steps": 5} # Expanded range

# For Gap Guardian, these are relevant for optimization
DEFAULT_ENTRY_START_HOUR_OPTIMIZATION_RANGE = {"min": 8, "max": 10, "steps": 3}
DEFAULT_ENTRY_START_MINUTE_OPTIMIZATION_VALUES = [0, 15, 30, 45]
DEFAULT_ENTRY_END_HOUR_OPTIMIZATION_RANGE = {"min": 10, "max": 12, "steps": 3}
DEFAULT_ENTRY_END_MINUTE_OPTIMIZATION_VALUES = [0] # Usually end of hour for GG

DEFAULT_RANDOM_SEARCH_ITERATIONS = 25
DEFAULT_WFO_IN_SAMPLE_DAYS = 90
DEFAULT_WFO_OUT_OF_SAMPLE_DAYS = 30
DEFAULT_WFO_STEP_DAYS = 30
MIN_TRADES_FOR_METRICS = 3
RISK_FREE_RATE = 0.01 # Annual risk-free rate for Sharpe/Sortino
TRADING_DAYS_PER_YEAR = 252 # For annualizing Sharpe/Sortino
