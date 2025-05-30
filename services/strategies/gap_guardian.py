# services/strategies/gap_guardian.py
"""
Signal generation logic for the Gap Guardian strategy.
"""
import pandas as pd
from datetime import time as dt_time
from utils.logger import get_logger # Assuming logger.py is in 'utils'
# from config import settings # Not directly used in this specific function, but good practice if defaults were needed

logger = get_logger(__name__)

def generate_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float,
    entry_start_time: dt_time,
    entry_end_time: dt_time
) -> pd.DataFrame:
    """
    Generates signals for the Gap Guardian strategy.
    Identifies the opening range and looks for false breakouts/breakdowns.
    """
    signals_list = []
    if data.empty:
        logger.warning("Gap Guardian: Input data is empty.")
        return pd.DataFrame()

    # Ensure data index is datetime (should be handled by data_loader)
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.error("Gap Guardian: Data index is not a DatetimeIndex.")
        return pd.DataFrame()
        
    # Group data by date to process each day independently
    for date_val, day_data in data.groupby(data.index.date):
        if day_data.empty:
            continue
        
        # Identify the opening bar based on entry_start_time
        # The first bar at or after entry_start_time is considered the opening bar for the range.
        opening_bar_candidates = day_data[day_data.index.time >= entry_start_time]
        if opening_bar_candidates.empty:
            # logger.debug(f"Gap Guardian ({date_val}): No bars on or after entry start time {entry_start_time}.")
            continue
        
        opening_bar_data = opening_bar_candidates.iloc[0:1] # Select the very first bar
        if opening_bar_data.empty: # Should not happen if opening_bar_candidates was not empty
            continue
            
        opening_bar_timestamp = opening_bar_data.index[0]
        opening_range_high = opening_bar_data['High'].iloc[0]
        opening_range_low = opening_bar_data['Low'].iloc[0]
        # logger.debug(f"Gap Guardian ({date_val}): Opening range [{opening_range_low:.2f} - {opening_range_high:.2f}] from bar at {opening_bar_timestamp.time()}")
        
        # Define the window for signal scanning: after the opening bar and before entry_end_time
        signal_scan_window_data = day_data[
            (day_data.index > opening_bar_timestamp) &  # Must be strictly after the opening bar
            (day_data.index.time < entry_end_time)     # And before the entry window closes
        ]
        
        if signal_scan_window_data.empty:
            # logger.debug(f"Gap Guardian ({date_val}): No bars in signal scan window.")
            continue

        # Iterate through bars in the scan window to find signals
        for idx, bar in signal_scan_window_data.iterrows():
            signal_time = idx # Timestamp of the current bar being evaluated for a signal

            # Long signal: Price dips below opening range low, then closes back above it
            if bar['Low'] < opening_range_low and bar['Close'] > opening_range_low:
                entry_price = bar['Close'] # Enter on close of the signal bar
                sl = entry_price - stop_loss_points
                tp = entry_price + (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time,
                    'SignalType': 'Long',
                    'EntryPrice': entry_price,
                    'SL': sl,
                    'TP': tp,
                    'Reason': f"GG Long: False breakdown of ORL {opening_range_low:.2f} at {bar.name.time()}"
                })
                # logger.info(f"Gap Guardian ({date_val}): Long signal at {signal_time}. Entry: {entry_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")
                break # Typically one signal per day for this strategy type

            # Short signal: Price pops above opening range high, then closes back below it
            elif bar['High'] > opening_range_high and bar['Close'] < opening_range_high:
                entry_price = bar['Close'] # Enter on close of the signal bar
                sl = entry_price + stop_loss_points
                tp = entry_price - (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time,
                    'SignalType': 'Short',
                    'EntryPrice': entry_price,
                    'SL': sl,
                    'TP': tp,
                    'Reason': f"GG Short: False breakout of ORH {opening_range_high:.2f} at {bar.name.time()}"
                })
                # logger.info(f"Gap Guardian ({date_val}): Short signal at {signal_time}. Entry: {entry_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")
                break # Typically one signal per day

    return pd.DataFrame(signals_list)
