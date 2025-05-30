# services/database_manager.py
"""
Manages all database interactions for caching market data and storing results.
Uses SQLite for simplicity and portability.
Includes indexing for improved query performance.
Database connection is cached using Streamlit's cache_resource and configured for multi-thread access.
"""
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
import os
import streamlit as st

from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

DB_FILE_PATH = "data/ict_strategies_app.db"

@st.cache_resource(show_spinner=False)
def get_db_connection():
    """
    Establishes and returns a database connection.
    The connection is cached using st.cache_resource for efficiency.
    Configured with check_same_thread=False to allow usage across Streamlit's threads.
    Ensures the 'data' directory for the database file exists.
    """
    logger.info(f"Attempting to establish database connection to: {DB_FILE_PATH}")
    try:
        db_dir = os.path.dirname(DB_FILE_PATH)
        if db_dir and not os.path.exists(db_dir): # Ensure db_dir is not empty string
            os.makedirs(db_dir)
            logger.info(f"Created database directory: {db_dir}")
        
        # Important: check_same_thread=False allows the connection to be used across threads,
        # which is necessary with Streamlit's @st.cache_resource and execution model.
        conn = sqlite3.connect(
            DB_FILE_PATH,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False # Allow connection usage across threads
        )
        conn.row_factory = sqlite3.Row
        logger.info(f"Database connection to {DB_FILE_PATH} established successfully (check_same_thread=False).")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error to {DB_FILE_PATH}: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not connect to the application database: {e}. Please check logs and ensure write permissions for the 'data' directory.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while establishing DB connection: {e}", exc_info=True)
        st.error(f"Fatal Error: An unexpected issue occurred with the database setup: {e}.")
        raise


def init_db():
    """
    Initializes the database and creates tables and indexes if they don't exist.
    Uses the cached database connection.
    """
    try:
        conn = get_db_connection()
        with conn:
            cursor = conn.cursor()

            # Market Data Cache Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_cache (
                ticker TEXT NOT NULL,
                interval TEXT NOT NULL,
                data_timestamp TIMESTAMP NOT NULL,
                Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER,
                PRIMARY KEY (ticker, interval, data_timestamp)
            )
            """)
            logger.debug("Table 'market_data_cache' checked/created.")
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_ticker_interval_ts 
            ON market_data_cache (ticker, interval, data_timestamp)
            """)
            logger.debug("Index 'idx_market_data_ticker_interval_ts' for 'market_data_cache' checked/created.")

            # Backtest Runs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy_name TEXT NOT NULL, ticker TEXT NOT NULL, timeframe TEXT NOT NULL,
                start_date DATE NOT NULL, end_date DATE NOT NULL,   
                initial_capital REAL, risk_per_trade_percent REAL,
                parameters TEXT, source TEXT, 
                performance_metrics TEXT, equity_curve TEXT 
            )
            """)
            logger.debug("Table 'backtest_runs' checked/created.")
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_backtest_runs_lookup 
            ON backtest_runs (strategy_name, ticker, timeframe, run_timestamp DESC)
            """)
            logger.debug("Index 'idx_backtest_runs_lookup' for 'backtest_runs' checked/created.")

            # Trades Log Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades_log (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, 
                EntryTime TIMESTAMP, EntryPrice REAL, Type TEXT, SL REAL, TP REAL,
                PositionSize REAL, ExitTime TIMESTAMP, ExitPrice REAL, P_L REAL, ExitReason TEXT,
                FOREIGN KEY (run_id) REFERENCES backtest_runs (run_id) ON DELETE CASCADE
            )
            """)
            logger.debug("Table 'trades_log' checked/created.")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_log_run_id ON trades_log (run_id)")
            logger.debug("Index 'idx_trades_log_run_id' for 'trades_log' checked/created.")

            # Optimization Results Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_results (
                opt_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                opt_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy_name TEXT NOT NULL, ticker TEXT NOT NULL, timeframe TEXT NOT NULL,
                start_date DATE NOT NULL, end_date DATE NOT NULL,   
                optimization_algorithm TEXT, optimized_metric TEXT,
                results_dataframe TEXT, extra_config TEXT DEFAULT NULL 
            )
            """)
            logger.debug("Table 'optimization_results' checked/created.")
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_optimization_results_lookup 
            ON optimization_results (strategy_name, ticker, timeframe, opt_timestamp DESC)
            """)
            logger.debug("Index 'idx_optimization_results_lookup' for 'optimization_results' checked/created.")
        logger.info(f"Database '{DB_FILE_PATH}' initialized successfully with tables and indexes.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database schema: {e}", exc_info=True)
        st.error(f"Critical Error: Could not initialize the database schema: {e}. Application database features may be unavailable.")
    except Exception as e: 
        logger.error(f"An unexpected error occurred during DB schema initialization: {e}", exc_info=True)
        st.error(f"Critical Error: An unexpected issue occurred during database schema setup: {e}.")


def save_market_data(ticker: str, interval: str, data_df: pd.DataFrame):
    if data_df.empty:
        logger.debug(f"Market data for {ticker} ({interval}) is empty, not saving to cache.")
        return
    if not isinstance(data_df.index, pd.DatetimeIndex):
        logger.error(f"Market data for {ticker} ({interval}) has non-DatetimeIndex. Cannot save.")
        return
    
    df_to_save = data_df.copy()
    try:
        if df_to_save.index.tz is None:
            df_to_save.index = df_to_save.index.tz_localize(settings.NY_TIMEZONE_STR)
        elif str(df_to_save.index.tz) != settings.NY_TIMEZONE_STR:
            df_to_save.index = df_to_save.index.tz_convert(settings.NY_TIMEZONE_STR)
    except Exception as e:
        logger.error(f"Failed to standardize timezone to NY for {ticker} ({interval}): {e}. Skipping cache save.", exc_info=True)
        return

    df_to_save = df_to_save.reset_index().rename(columns={'index': 'data_timestamp', 'Datetime': 'data_timestamp'}, errors='ignore')
    if 'data_timestamp' not in df_to_save.columns:
        logger.error(f"Column 'data_timestamp' missing for {ticker} ({interval}). Cannot save.")
        return

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df_to_save.columns:
            df_to_save[col] = np.nan if col != 'Volume' else 0
    df_to_save['ticker'] = ticker
    df_to_save['interval'] = interval
    db_cols = ['ticker', 'interval', 'data_timestamp'] + required_cols
    df_to_save = df_to_save[db_cols]

    try:
        conn = get_db_connection()
        with conn:
            min_ts_str = pd.Timestamp(df_to_save['data_timestamp'].min()).isoformat()
            max_ts_str = pd.Timestamp(df_to_save['data_timestamp'].max()).isoformat()
            delete_query = "DELETE FROM market_data_cache WHERE ticker = ? AND interval = ? AND data_timestamp BETWEEN ? AND ?"
            conn.execute(delete_query, (ticker, interval, min_ts_str, max_ts_str))
            df_to_save.to_sql('market_data_cache', conn, if_exists='append', index=False)
            logger.info(f"Saved/Updated {len(df_to_save)} rows for {ticker} ({interval}) to cache.")
    except Exception as e:
        logger.error(f"Error saving market data for {ticker} ({interval}): {e}", exc_info=True)

def load_market_data(ticker: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        conn = get_db_connection()
        start_dt_ny_iso = pd.Timestamp(start_date, tz=settings.NY_TIMEZONE_STR).isoformat()
        # For end_date, ensure it covers the entire day by going to the very end of that day
        end_dt_ny_iso = (pd.Timestamp(datetime.combine(end_date, dt_time.max), tz=settings.NY_TIMEZONE_STR)).isoformat()

        query = """
        SELECT data_timestamp, Open, High, Low, Close, Volume 
        FROM market_data_cache
        WHERE ticker = ? AND interval = ? AND data_timestamp BETWEEN ? AND ?
        ORDER BY data_timestamp ASC
        """
        df = pd.read_sql_query(query, conn, params=(ticker, interval, start_dt_ny_iso, end_dt_ny_iso))

        if not df.empty:
            df['data_timestamp'] = pd.to_datetime(df['data_timestamp']).dt.tz_convert(settings.NY_TIMEZONE_STR)
            df.set_index('data_timestamp', inplace=True)
            
            user_req_start_ts_ny = pd.Timestamp(start_date, tz=settings.NY_TIMEZONE_STR)
            user_req_end_ts_ny_day_start = pd.Timestamp(end_date, tz=settings.NY_TIMEZONE_STR)

            if df.index.min() <= user_req_start_ts_ny and df.index.max().normalize() >= user_req_end_ts_ny_day_start.normalize():
                logger.info(f"Loaded {len(df)} rows for {ticker} ({interval}) from cache.")
                return df
            else:
                logger.info(f"Cached data for {ticker} ({interval}) does not fully cover requested range.")
        else:
            logger.info(f"No data in cache for {ticker} ({interval}) for range {start_dt_ny_iso} to {end_dt_ny_iso}.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading market data for {ticker} ({interval}): {e}", exc_info=True)
        return pd.DataFrame()

def save_backtest_results(
    strategy_name: str, ticker: str, timeframe: str,
    start_date_dt: date, end_date_dt: date,
    initial_capital: float, risk_per_trade_percent: float,
    parameters: dict, source: str,
    performance_metrics: dict, trades_df: pd.DataFrame, equity_curve_series: pd.Series
    ) -> int | None:
    try:
        conn = get_db_connection()
        with conn:
            cursor = conn.cursor()
            params_json = json.dumps(parameters, default=str)
            perf_metrics_json = json.dumps(performance_metrics, default=str)
            equity_items = {ts.isoformat(): val for ts, val in equity_curve_series.items()} if not equity_curve_series.empty else {}
            equity_curve_json = json.dumps(equity_items)

            cursor.execute("""
            INSERT INTO backtest_runs (strategy_name, ticker, timeframe, start_date, end_date, initial_capital, risk_per_trade_percent, parameters, source, performance_metrics, equity_curve)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (strategy_name, ticker, timeframe, start_date_dt.isoformat(), end_date_dt.isoformat(), initial_capital, risk_per_trade_percent, params_json, source, perf_metrics_json, equity_curve_json))
            run_id = cursor.lastrowid
            
            if run_id and not trades_df.empty:
                trades_to_save = trades_df.copy()
                trades_to_save['run_id'] = run_id
                for col in ['EntryTime', 'ExitTime']:
                    if col in trades_to_save.columns:
                        trades_to_save[col] = pd.to_datetime(trades_to_save[col]).map(lambda x: x.isoformat() if pd.notnull(x) else None)
                trades_to_save.rename(columns={'P&L': 'P_L'}, inplace=True)
                trade_db_cols = ['run_id', 'EntryTime', 'EntryPrice', 'Type', 'SL', 'TP', 'PositionSize', 'ExitTime', 'ExitPrice', 'P_L', 'ExitReason']
                trades_to_save = trades_to_save.reindex(columns=trade_db_cols) # Ensure all columns exist
                trades_to_save.to_sql('trades_log', conn, if_exists='append', index=False)
            logger.info(f"Backtest results for '{strategy_name}' on '{ticker}' saved with run_id: {run_id}")
            return run_id
    except Exception as e:
        logger.error(f"Error saving backtest results: {e}", exc_info=True)
    return None

def save_optimization_results(
    strategy_name: str, ticker: str, timeframe: str,
    start_date_dt: date, end_date_dt: date,
    optimization_algorithm: str, optimized_metric: str,
    results_df: pd.DataFrame, extra_config: dict | None = None 
    ):
    if results_df.empty:
        logger.info("Optimization results DataFrame empty, not saving.")
        return
    try:
        conn = get_db_connection()
        with conn:
            results_json = results_df.to_json(orient='records', date_format='iso', default_handler=str)
            extra_config_json = json.dumps(extra_config, default=str) if extra_config else None
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO optimization_results (strategy_name, ticker, timeframe, start_date, end_date, optimization_algorithm, optimized_metric, results_dataframe, extra_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (strategy_name, ticker, timeframe, start_date_dt.isoformat(), end_date_dt.isoformat(), optimization_algorithm, optimized_metric, results_json, extra_config_json))
            opt_run_id = cursor.lastrowid
            logger.info(f"Optimization results for '{strategy_name}' on '{ticker}' saved with opt_run_id: {opt_run_id}.")
    except Exception as e:
        logger.error(f"Error saving optimization results: {e}", exc_info=True)

if __name__ == '__main__':
    logger.info("Running database_manager.py directly for testing DB initialization.")
    init_db()
    logger.info("Database manager test complete.")
