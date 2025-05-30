# services/database_manager.py
"""
Manages all database interactions for caching market data and storing results.
Uses SQLite for simplicity and portability.
Includes indexing for improved query performance.
"""
import sqlite3
import pandas as pd
import numpy as np # Added for handling np.nan if necessary
import json # For storing dicts like performance metrics
from datetime import datetime, date
import os # For ensuring data directory exists

from utils.logger import get_logger # Assuming logger.py is in 'utils'
from config import settings # For NY_TIMEZONE_STR

logger = get_logger(__name__)

DB_FILE_PATH = "data/ict_strategies_app.db" # Store DB in a 'data' subdirectory

def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        # Ensure the 'data' directory exists
        os.makedirs(os.path.dirname(DB_FILE_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DB_FILE_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error to {DB_FILE_PATH}: {e}", exc_info=True)
        raise # Re-raise the exception to be handled by the caller

def init_db():
    """Initializes the database and creates tables and indexes if they don't exist."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Market Data Cache Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_cache (
                ticker TEXT NOT NULL,
                interval TEXT NOT NULL,
                data_timestamp TIMESTAMP NOT NULL, -- Stored as TEXT by pandas if not careful, ensure it's recognized as TIMESTAMP
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER,
                PRIMARY KEY (ticker, interval, data_timestamp)
            )
            """)
            logger.info("Table 'market_data_cache' checked/created.")
            # Index for market_data_cache
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_ticker_interval_ts 
            ON market_data_cache (ticker, interval, data_timestamp)
            """)
            logger.info("Index 'idx_market_data_ticker_interval_ts' for 'market_data_cache' checked/created.")

            # Backtest Runs Table (Summary)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy_name TEXT NOT NULL,
                ticker TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                start_date DATE NOT NULL, -- Stored as TEXT
                end_date DATE NOT NULL,   -- Stored as TEXT
                initial_capital REAL,
                risk_per_trade_percent REAL,
                parameters TEXT, -- JSON string of specific params
                source TEXT, -- e.g., 'Manual', 'Optimization', 'WFO'
                performance_metrics TEXT, -- JSON string of performance dict
                equity_curve TEXT -- JSON string of equity curve (timestamp: value)
            )
            """)
            logger.info("Table 'backtest_runs' checked/created.")
            # Index for backtest_runs
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_backtest_runs_lookup 
            ON backtest_runs (strategy_name, ticker, timeframe, run_timestamp DESC)
            """) # run_timestamp DESC for fetching recent runs faster
            logger.info("Index 'idx_backtest_runs_lookup' for 'backtest_runs' checked/created.")


            # Trades Log Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades_log (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL, -- Foreign key to backtest_runs
                EntryTime TIMESTAMP,
                EntryPrice REAL,
                Type TEXT, -- 'Long' or 'Short'
                SL REAL,
                TP REAL,
                PositionSize REAL,
                ExitTime TIMESTAMP,
                ExitPrice REAL,
                P_L REAL, -- P&L for the trade
                ExitReason TEXT,
                FOREIGN KEY (run_id) REFERENCES backtest_runs (run_id) ON DELETE CASCADE
            )
            """)
            logger.info("Table 'trades_log' checked/created.")
            # Index for trades_log
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_log_run_id 
            ON trades_log (run_id)
            """)
            logger.info("Index 'idx_trades_log_run_id' for 'trades_log' checked/created.")

            # Optimization Results Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_results (
                opt_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                opt_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy_name TEXT NOT NULL,
                ticker TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                start_date DATE NOT NULL, -- Stored as TEXT
                end_date DATE NOT NULL,   -- Stored as TEXT
                optimization_algorithm TEXT,
                optimized_metric TEXT,
                results_dataframe TEXT, -- JSON string of the entire optimization results DataFrame
                extra_config TEXT DEFAULT NULL -- JSON string for additional config like AI settings
            )
            """)
            logger.info("Table 'optimization_results' checked/created.")
             # Index for optimization_results
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_optimization_results_lookup 
            ON optimization_results (strategy_name, ticker, timeframe, opt_timestamp DESC)
            """)
            logger.info("Index 'idx_optimization_results_lookup' for 'optimization_results' checked/created.")
            
            conn.commit()
        logger.info(f"Database '{DB_FILE_PATH}' initialized successfully with tables and indexes.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
    except Exception as e: # Catch any other unexpected error
        logger.error(f"An unexpected error occurred during DB initialization: {e}", exc_info=True)


# --- Market Data Cache Functions ---
def save_market_data(ticker: str, interval: str, data_df: pd.DataFrame):
    """Saves market data to the cache. Expects data_df index to be DatetimeIndex."""
    if data_df.empty:
        logger.debug(f"Market data for {ticker} ({interval}) is empty, not saving to cache.")
        return

    if not isinstance(data_df.index, pd.DatetimeIndex):
        logger.error(f"Market data for {ticker} ({interval}) has non-DatetimeIndex. Cannot save to cache.")
        return
    
    df_to_save = data_df.copy()
    # Standardize timezone to NY before saving
    if df_to_save.index.tz is None:
        logger.warning(f"Market data for {ticker} ({interval}) is timezone-naive. Localizing to NY before saving.")
        try:
            df_to_save.index = df_to_save.index.tz_localize(settings.NY_TIMEZONE_STR)
        except Exception as e:
            logger.error(f"Failed to localize market data index to NY for {ticker} ({interval}): {e}. Trying UTC then NY.", exc_info=True)
            try:
                df_to_save.index = df_to_save.index.tz_localize('UTC').tz_convert(settings.NY_TIMEZONE_STR)
            except Exception as e_utc:
                logger.error(f"Failed to convert market data index via UTC to NY for {ticker} ({interval}): {e_utc}. Skipping cache save.", exc_info=True)
                return
    elif str(df_to_save.index.tz) != settings.NY_TIMEZONE_STR:
        logger.warning(f"Market data for {ticker} ({interval}) has timezone {df_to_save.index.tz}. Converting to NY before saving.")
        try:
            df_to_save.index = df_to_save.index.tz_convert(settings.NY_TIMEZONE_STR)
        except Exception as e:
            logger.error(f"Failed to convert market data index to NY for {ticker} ({interval}): {e}. Skipping cache save.", exc_info=True)
            return

    df_to_save = df_to_save.reset_index()
    # Standardize timestamp column name
    df_to_save.rename(columns={'index': 'data_timestamp', 'Datetime': 'data_timestamp', 'Timestamp': 'data_timestamp'}, inplace=True, errors='ignore')
    
    if 'data_timestamp' not in df_to_save.columns:
        logger.error(f"Critical column 'data_timestamp' missing after processing for {ticker} ({interval}). Columns: {df_to_save.columns}. Cannot save.")
        return

    required_cols = ['data_timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df_to_save.columns:
            df_to_save[col] = np.nan if col != 'Volume' else 0
            logger.warning(f"Column '{col}' was missing for {ticker} ({interval}). Added with default values.")

    df_to_save['ticker'] = ticker
    df_to_save['interval'] = interval
    db_cols = ['ticker', 'interval', 'data_timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_to_save = df_to_save[db_cols]
    # Ensure data_timestamp is in a format SQLite understands as TIMESTAMP for comparisons
    # Pandas to_sql usually handles datetime64[ns, TZ] to ISO8601 strings, which SQLite can compare.
    # df_to_save['data_timestamp'] = df_to_save['data_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f') # Explicit conversion if issues

    try:
        with get_db_connection() as conn:
            if not df_to_save.empty:
                min_ts = df_to_save['data_timestamp'].min()
                max_ts = df_to_save['data_timestamp'].max()
                # Convert Timestamps to string for SQL query if they are not already
                # This is important as SQLite might treat raw Timestamp objects differently than ISO strings
                min_ts_str = pd.Timestamp(min_ts).isoformat()
                max_ts_str = pd.Timestamp(max_ts).isoformat()

                delete_query = """
                DELETE FROM market_data_cache 
                WHERE ticker = ? AND interval = ? AND data_timestamp BETWEEN ? AND ?
                """
                conn.execute(delete_query, (ticker, interval, min_ts_str, max_ts_str))
                df_to_save.to_sql('market_data_cache', conn, if_exists='append', index=False)
                conn.commit()
                logger.info(f"Saved/Updated {len(df_to_save)} rows of market data for {ticker} ({interval}) to cache. Range: {min_ts} to {max_ts}")
    except sqlite3.Error as e:
        logger.error(f"Error saving market data for {ticker} ({interval}) to cache: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error saving market data for {ticker} ({interval}): {e}", exc_info=True)


def load_market_data(ticker: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Loads market data from cache for the given ticker, interval, and date range (inclusive)."""
    try:
        with get_db_connection() as conn:
            # Convert start_date and end_date to NY-aware timestamps for query
            start_dt_ny = pd.Timestamp(start_date, tz=settings.NY_TIMEZONE_STR).isoformat()
            end_dt_ny = (pd.Timestamp(end_date, tz=settings.NY_TIMEZONE_STR) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).isoformat()

            query = """
            SELECT data_timestamp, Open, High, Low, Close, Volume 
            FROM market_data_cache
            WHERE ticker = ? AND interval = ? AND data_timestamp BETWEEN ? AND ?
            ORDER BY data_timestamp ASC
            """
            df = pd.read_sql_query(query, conn, params=(ticker, interval, start_dt_ny, end_dt_ny))

        if not df.empty:
            df['data_timestamp'] = pd.to_datetime(df['data_timestamp']) # Convert from string if needed
            # Ensure timezone is NYT, as it's stored that way (or should be)
            if df['data_timestamp'].dt.tz is None:
                 df['data_timestamp'] = df['data_timestamp'].dt.tz_localize('UTC').dt.tz_convert(settings.NY_TIMEZONE_STR) # Assume UTC if naive
            elif str(df['data_timestamp'].dt.tz) != settings.NY_TIMEZONE_STR:
                 df['data_timestamp'] = df['data_timestamp'].dt.tz_convert(settings.NY_TIMEZONE_STR)

            df.set_index('data_timestamp', inplace=True)
            
            # Verify if the loaded data fully covers the requested range
            user_req_start_ts_ny = pd.Timestamp(start_date, tz=settings.NY_TIMEZONE_STR)
            user_req_end_ts_ny_day_start = pd.Timestamp(end_date, tz=settings.NY_TIMEZONE_STR)

            if not df.empty and df.index.min() <= user_req_start_ts_ny and \
               df.index.max().normalize() >= user_req_end_ts_ny_day_start.normalize():
                logger.info(f"Loaded {len(df)} rows of market data for {ticker} ({interval}) from cache. Range: {df.index.min()} to {df.index.max()}")
                return df
            else:
                logger.info(f"Market data for {ticker} ({interval}) in cache (found {len(df)} rows) does not fully cover requested range {user_req_start_ts_ny} to {user_req_end_ts_ny_day_start}. Min cache: {df.index.min() if not df.empty else 'N/A'}, Max cache: {df.index.max() if not df.empty else 'N/A'}")
        else:
            logger.info(f"No market data found in cache for {ticker} ({interval}) for range {start_dt_ny} to {end_dt_ny}.")
        
        return pd.DataFrame() # Return empty if not fully covered or not found
            
    except sqlite3.Error as e:
        logger.error(f"Error loading market data for {ticker} ({interval}) from cache: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading market data for {ticker} ({interval}): {e}", exc_info=True)
        return pd.DataFrame()

# --- Results Storage Functions ---
def save_backtest_results(
    strategy_name: str, ticker: str, timeframe: str,
    start_date_dt: date, end_date_dt: date,
    initial_capital: float, risk_per_trade_percent: float,
    parameters: dict, source: str,
    performance_metrics: dict, trades_df: pd.DataFrame, equity_curve_series: pd.Series
    ) -> int | None:
    """Saves backtest summary, trades, and equity curve to the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            params_json = json.dumps(parameters, default=str)
            perf_metrics_json = json.dumps(performance_metrics, default=str)
            equity_curve_dict = {ts.isoformat(): val for ts, val in equity_curve_series.items()} if not equity_curve_series.empty else {}
            equity_curve_json = json.dumps(equity_curve_dict)

            cursor.execute("""
            INSERT INTO backtest_runs 
            (strategy_name, ticker, timeframe, start_date, end_date, initial_capital, risk_per_trade_percent, parameters, source, performance_metrics, equity_curve)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_name, ticker, timeframe,
                start_date_dt.isoformat(), end_date_dt.isoformat(),
                initial_capital, risk_per_trade_percent,
                params_json, source, perf_metrics_json, equity_curve_json
            ))
            run_id = cursor.lastrowid
            
            if run_id and not trades_df.empty:
                trades_to_save = trades_df.copy()
                trades_to_save['run_id'] = run_id
                for col in ['EntryTime', 'ExitTime']:
                    if col in trades_to_save.columns:
                        # Ensure datetime objects are converted to ISO format strings for SQLite
                        trades_to_save[col] = pd.to_datetime(trades_to_save[col]).map(lambda x: x.isoformat() if pd.notnull(x) else None)

                trades_to_save.rename(columns={'P&L': 'P_L'}, inplace=True)
                trade_db_cols = ['run_id', 'EntryTime', 'EntryPrice', 'Type', 'SL', 'TP', 
                                 'PositionSize', 'ExitTime', 'ExitPrice', 'P_L', 'ExitReason']
                
                # Ensure all columns exist, add NaNs if not
                for tc in trade_db_cols:
                    if tc not in trades_to_save.columns:
                        trades_to_save[tc] = np.nan 
                trades_to_save = trades_to_save[trade_db_cols]
                trades_to_save.to_sql('trades_log', conn, if_exists='append', index=False)
            
            conn.commit()
            logger.info(f"Backtest results for strategy '{strategy_name}' on '{ticker}' saved with run_id: {run_id}")
            return run_id
    except sqlite3.Error as e:
        logger.error(f"Error saving backtest results: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error saving backtest results: {e}", exc_info=True)
    return None

def save_optimization_results(
    strategy_name: str, ticker: str, timeframe: str,
    start_date_dt: date, end_date_dt: date,
    optimization_algorithm: str, optimized_metric: str,
    results_df: pd.DataFrame,
    extra_config: dict | None = None # Added for AI settings etc.
    ):
    """Saves optimization results DataFrame to the database."""
    if results_df.empty:
        logger.info("Optimization results DataFrame is empty, not saving.")
        return
    try:
        with get_db_connection() as conn:
            results_json = results_df.to_json(orient='records', date_format='iso', default_handler=str)
            extra_config_json = json.dumps(extra_config, default=str) if extra_config else None

            conn.execute("""
            INSERT INTO optimization_results
            (strategy_name, ticker, timeframe, start_date, end_date, optimization_algorithm, optimized_metric, results_dataframe, extra_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_name, ticker, timeframe,
                start_date_dt.isoformat(), end_date_dt.isoformat(),
                optimization_algorithm, optimized_metric, results_json, extra_config_json
            ))
            conn.commit()
            opt_run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            logger.info(f"Optimization results for strategy '{strategy_name}' on '{ticker}' saved with opt_run_id: {opt_run_id}. Extra config: {extra_config_json is not None}")
    except sqlite3.Error as e:
        logger.error(f"Error saving optimization results: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error saving optimization results: {e}", exc_info=True)

if __name__ == '__main__':
    logger.info("Running database_manager.py directly for testing DB initialization.")
    init_db()
    logger.info("Database manager test run complete. Check for 'ict_strategies_app.db' in 'data/' directory and log messages for table/index creation.")

    # Example: Test saving optimization results with extra_config
    # mock_opt_df_test = pd.DataFrame({
    #     'SL Points': [10, 15], 'RRR': [2, 2.5], 'Total P&L': [1000, 1200]
    # })
    # mock_extra_config_test = {"AI_Filter_Enabled": True, "AI_Model": "RandomForest"}
    # save_optimization_results("TestStrategy", "TESTTICKER", "15m", date(2023,1,1), date(2023,1,1), 
    #                           "Grid Search", "Total P&L", mock_opt_df_test, mock_extra_config_test)
