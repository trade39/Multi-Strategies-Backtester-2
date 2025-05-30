# services/strategies/silver_bullet.py
"""
Signal generation logic for the Silver Bullet strategy (Time-based FVG).
"""
import pandas as pd
from config import settings # For SILVER_BULLET_WINDOWS_NY and NY_TIMEZONE
from utils.logger import get_logger
from .technical_utils import find_fvg # Import from local package

logger = get_logger(__name__)

def generate_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    """
    Generates signals for the Silver Bullet strategy.
    Logic: FVG entry within specific 1-hour NY time windows.
    """
    signals_list = []
    if data.empty:
        logger.warning("Silver Bullet: Input data is empty.")
        return pd.DataFrame()

    # Ensure data index is datetime and NY localized (should be by data_loader)
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz != settings.NY_TIMEZONE:
        logger.error(f"Silver Bullet: Data must have NY-localized DatetimeIndex. Current tz: {data.index.tz}")
        # Attempt conversion if not NYT, though data_loader should handle this.
        try:
            if data.index.tz is None: data = data.tz_localize('UTC').tz_convert(settings.NY_TIMEZONE_STR)
            else: data = data.tz_convert(settings.NY_TIMEZONE_STR)
            logger.info(f"Silver Bullet: Converted data to NYT. Index tz is now: {data.index.tz}")
        except Exception as e:
            logger.error(f"Silver Bullet: Failed to convert data to NYT. Error: {e}")
            return pd.DataFrame()

    # Iterate, leaving room for FVG pattern (i-3, i-2, i-1) for entry on bar 'i'
    for i in range(3, len(data)):
        current_bar = data.iloc[i]
        current_bar_time_obj = data.index[i].time() # Get the time object for comparison
        current_bar_timestamp = data.index[i] # Full timestamp for signal log

        # Check if current bar's time is within any Silver Bullet window
        in_sb_window = False
        for start_t, end_t in settings.SILVER_BULLET_WINDOWS_NY:
            if start_t <= current_bar_time_obj < end_t:
                in_sb_window = True
                break
        
        if not in_sb_window:
            continue # Skip if not in a designated Silver Bullet window

        # --- Bullish FVG Entry ---
        # FVG formed by (i-3, i-2, i-1); entry on current_bar 'i'
        bullish_fvg_zone = find_fvg(data, i - 3, "bullish")
        if bullish_fvg_zone:
            fvg_low, fvg_high = bullish_fvg_zone
            # Entry: current bar dips into FVG, closes bullishly above FVG low
            if current_bar['Low'] <= fvg_high and \
               current_bar['Close'] > fvg_low and \
               current_bar['Close'] > current_bar['Open']:
                
                entry_price = current_bar['Close']
                sl = entry_price - stop_loss_points
                tp = entry_price + (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': current_bar_timestamp,
                    'SignalType': 'Long',
                    'EntryPrice': entry_price,
                    'SL': sl,
                    'TP': tp,
                    'Reason': f"SB Long: Retrace to FVG({fvg_low:.2f}-{fvg_high:.2f}) in window at {current_bar_timestamp.time()}"
                })
                # logger.info(f"SB Long @ {current_bar_timestamp}, FVG: {fvg_low:.2f}-{fvg_high:.2f}, Entry: {entry_price:.2f}")
                # To avoid multiple signals from the same FVG within the same window, could add a break or more complex state.
                # For now, allow if conditions re-trigger on subsequent bars in window.
                # continue # Process next bar, even if signal found, to catch all FVG entries in window.


        # --- Bearish FVG Entry ---
        # FVG formed by (i-3, i-2, i-1); entry on current_bar 'i'
        bearish_fvg_zone = find_fvg(data, i - 3, "bearish")
        if bearish_fvg_zone:
            fvg_low, fvg_high = bearish_fvg_zone
            # Entry: current bar touches FVG, closes bearishly below FVG high
            if current_bar['High'] >= fvg_low and \
               current_bar['Close'] < fvg_high and \
               current_bar['Close'] < current_bar['Open']:
                
                entry_price = current_bar['Close']
                sl = entry_price + stop_loss_points
                tp = entry_price - (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': current_bar_timestamp,
                    'SignalType': 'Short',
                    'EntryPrice': entry_price,
                    'SL': sl,
                    'TP': tp,
                    'Reason': f"SB Short: Retrace to FVG({fvg_low:.2f}-{fvg_high:.2f}) in window at {current_bar_timestamp.time()}"
                })
                # logger.info(f"SB Short @ {current_bar_timestamp}, FVG: {fvg_low:.2f}-{fvg_high:.2f}, Entry: {entry_price:.2f}")
                # continue

    if not signals_list:
        logger.info("Silver Bullet: No signals generated.")
        
    return pd.DataFrame(signals_list)
