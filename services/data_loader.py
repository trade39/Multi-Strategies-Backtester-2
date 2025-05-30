# services/data_loader.py
"""
Handles fetching and preparing market data using yfinance,
with an integrated SQLite cache via database_manager.
"""
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, date
from config.settings import (
    MAX_SHORT_INTRADAY_DAYS, MAX_HOURLY_INTRADAY_DAYS,
    NY_TIMEZONE_STR, NY_TIMEZONE,
    YFINANCE_SHORT_INTRADAY_INTERVALS, YFINANCE_HOURLY_INTERVALS
)
from utils.logger import get_logger
from . import database_manager # Import the database manager

logger = get_logger(__name__)

# @st.cache_data(ttl=3600) # Streamlit caching can be layered on top of DB caching if needed
def fetch_historical_data(ticker: str, start_date_input: date, end_date_input: date, interval: str) -> pd.DataFrame:
    logger.info(f"Requesting data for {ticker} ({interval}) from {start_date_input} to {end_date_input}.")

    # 1. Try to load from database cache first
    # Ensure start_date_input and end_date_input are datetime objects for load_market_data
    start_datetime_for_cache = datetime.combine(start_date_input, datetime.min.time())
    end_datetime_for_cache = datetime.combine(end_date_input, datetime.max.time()) # Ensure full end day

    cached_data = database_manager.load_market_data(ticker, interval, start_datetime_for_cache, end_datetime_for_cache)
    
    if not cached_data.empty:
        # Further check if the cached data truly covers the *exact* start and end dates requested by the user.
        # load_market_data already tries to ensure this, but an extra check here is fine.
        # Convert user input dates to NY-aware pandas Timestamps for precise comparison
        user_req_start_ts_ny = pd.Timestamp(start_date_input, tz=NY_TIMEZONE_STR)
        user_req_end_ts_ny_day_start = pd.Timestamp(end_date_input, tz=NY_TIMEZONE_STR)

        if cached_data.index.min() <= user_req_start_ts_ny and \
           cached_data.index.max().normalize() >= user_req_end_ts_ny_day_start.normalize(): # Compare normalized dates for end
            logger.info(f"Successfully loaded sufficient data for {ticker} ({interval}) from DB cache.")
            # Filter to exact user requested date range (day-level precision for start/end)
            # The cache might have more data than requested for that day.
            final_cached_data = cached_data[(cached_data.index >= user_req_start_ts_ny) & 
                                            (cached_data.index < (user_req_end_ts_ny_day_start + pd.Timedelta(days=1)))]
            if not final_cached_data.empty:
                return final_cached_data
            else:
                logger.info(f"Cached data for {ticker} ({interval}) existed but became empty after precise range filtering. Will fetch from yfinance.")
        else:
            logger.info(f"Cached data for {ticker} ({interval}) found but did not fully cover requested range. Will fetch from yfinance.")
            logger.debug(f"Cache range: {cached_data.index.min()} to {cached_data.index.max()}. Requested: {user_req_start_ts_ny} to {user_req_end_ts_ny_day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)}")


    # 2. If not in cache or insufficient, fetch from yfinance
    logger.info(f"Fetching data for {ticker} ({interval}) from yfinance API.")
    
    current_start_date_for_api = start_date_input # Use original user start date for API call logic

    if current_start_date_for_api >= end_date_input:
        logger.error(f"Initial validation failed: Start date {current_start_date_for_api} must be before end date {end_date_input} for {ticker}.")
        return pd.DataFrame()

    max_history_days = None
    if interval in YFINANCE_SHORT_INTRADAY_INTERVALS:
        max_history_days = MAX_SHORT_INTRADAY_DAYS
    elif interval in YFINANCE_HOURLY_INTERVALS:
        max_history_days = MAX_HOURLY_INTRADAY_DAYS

    if max_history_days is not None:
        # yfinance start date for intraday is based on "now"
        max_permissible_start_datetime_for_api = datetime.now(NY_TIMEZONE) - timedelta(days=max_history_days)
        max_permissible_start_date_for_api = max_permissible_start_datetime_for_api.date()
        if current_start_date_for_api < max_permissible_start_date_for_api:
            logger.warning(
                f"Requested start date {current_start_date_for_api} for {ticker} ({interval}) is too old for yfinance API. "
                f"Adjusting API fetch start to {max_permissible_start_date_for_api} (limit: ~{max_history_days} days)."
            )
            current_start_date_for_api = max_permissible_start_date_for_api

    if current_start_date_for_api >= end_date_input:
        logger.error(f"Date range validation failed after yfinance API adjustment for {ticker} ({interval}): "
                       f"Adjusted API start date {current_start_date_for_api} is not before end date {end_date_input}.")
        st.warning(f"The selected date range for {ticker} ({interval}) is invalid after adjusting for yfinance API data limits. "
                   f"Effective API start: {current_start_date_for_api.strftime('%Y-%m-%d')}, End: {end_date_input.strftime('%Y-%m-%d')}. "
                   f"Please select a valid range or expect limited data.")
        # We might still try to fetch what we can if current_start_date_for_api < end_date_input
        if current_start_date_for_api >= end_date_input: return pd.DataFrame()


    # yfinance 'end' parameter is exclusive, so add 1 day to include the end_date_input
    fetch_api_end_date = end_date_input + timedelta(days=1)

    try:
        logger.info(f"Proceeding to yf.download for {ticker}: API Start={current_start_date_for_api}, API End={fetch_api_end_date} (Interval: {interval}).")
        data = yf.download(
            tickers=ticker, start=current_start_date_for_api, end=fetch_api_end_date,
            interval=interval, progress=False, auto_adjust=False, actions=False,
            timeout=15 # Add a timeout
        )

        if data.empty:
            logger.warning(f"yf.download returned empty DataFrame for {ticker} (API Period: {current_start_date_for_api} to {fetch_api_end_date}, Interval: {interval}).")
            # Do not show st.warning here if cache was attempted, let app.py handle no data display
            return pd.DataFrame()

        logger.info(f"Downloaded data for {ticker}. Rows: {len(data)}. Initial yf columns: {data.columns.tolist()}")

        # --- Standard data processing ---
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        std_rename_map = {
            col_str: expected_name for col_str in data.columns
            for key, expected_name in {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close',
                                       'adjclose':'Adj Close', 'adj close':'Adj Close', 'volume':'Volume'}.items()
            if str(col_str).lower().replace(' ','').replace('.','') == key
        }
        if std_rename_map: data.rename(columns=std_rename_map, inplace=True)
        
        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in req_cols if col not in data.columns]
        if missing:
            logger.error(f"Data for {ticker} missing required columns after yf.download: {missing}. Available: {data.columns.tolist()}")
            return pd.DataFrame() # Critical columns missing

        if data.index.tz is None:
            try: data = data.tz_localize('UTC') # yfinance usually returns UTC for intraday
            except Exception as tz_err:
                logger.error(f"Could not localize naive DatetimeIndex to UTC for {ticker}: {tz_err}", exc_info=True)
                return pd.DataFrame()
        
        try: data = data.tz_convert(NY_TIMEZONE_STR)
        except Exception as tz_conv_err:
            logger.error(f"Could not convert timezone to {NY_TIMEZONE_STR} for {ticker}: {tz_conv_err}", exc_info=True)
            return pd.DataFrame()
        
        # Filter to the originally requested user date range *after* fetching potentially wider API range
        # This ensures we only cache and return what the user asked for, even if API gave more due to adjustments.
        user_start_datetime_inclusive = pd.Timestamp(start_date_input, tz=NY_TIMEZONE_STR)
        user_end_datetime_exclusive = pd.Timestamp(end_date_input, tz=NY_TIMEZONE_STR) + pd.Timedelta(days=1)
        
        data_filtered_to_user_request = data[
            (data.index >= user_start_datetime_inclusive) & 
            (data.index < user_end_datetime_exclusive)
        ]
        
        if data_filtered_to_user_request.empty:
            logger.warning(f"Data for {ticker} became empty after filtering to user's precise date range: {start_date_input} to {end_date_input}.")
            return pd.DataFrame()
            
        data_filtered_to_user_request.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # Use inplace with caution or assign back
        if data_filtered_to_user_request.empty:
            logger.warning(f"Data for {ticker} became empty after dropna on user-filtered range.")
            return pd.DataFrame()
        
        # 3. Save the newly fetched and processed data to cache
        # Pass the data that corresponds to the user's original request for caching.
        database_manager.save_market_data(ticker, interval, data_filtered_to_user_request.copy()) # Save a copy
            
        logger.info(f"Successfully fetched, processed, and cached {len(data_filtered_to_user_request)} rows for {ticker} ({interval}). Range: {data_filtered_to_user_request.index.min()} to {data_filtered_to_user_request.index.max()}).")
        return data_filtered_to_user_request

    except Exception as e:
        logger.error(f"General error in fetch_historical_data for {ticker} ({interval}): {e}", exc_info=True)
        # Do not show st.error here, let app.py handle it based on empty DataFrame
        return pd.DataFrame()

