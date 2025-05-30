# services/strategies/unicorn.py
"""
Signal generation logic for the Unicorn strategy (Breaker + FVG Overlap).
"""
import pandas as pd
from config import settings # For UNICORN_SWING_LOOKBACK, NY_TIMEZONE
from utils.logger import get_logger
from .technical_utils import find_swing_points, find_fvg, find_breaker_structures

logger = get_logger(__name__)

def generate_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    """
    Generates signals for the Unicorn strategy by identifying:
    1. A liquidity sweep of a prior swing point.
    2. A Market Structure Shift (MSS) breaking another swing point.
    3. A Breaker Block (last counter-trend candle(s) before MSS).
    4. A Fair Value Gap (FVG) created during the MSS displacement.
    5. Overlap between the Breaker Block and the FVG.
    Entry is on retracement into this overlapping zone.
    """
    signals_list = []
    if data.empty:
        logger.warning("Unicorn: Input data is empty.")
        return pd.DataFrame()

    # Ensure data index is datetime and NY localized (should be by data_loader)
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz != settings.NY_TIMEZONE:
        logger.error(f"Unicorn: Data must have NY-localized DatetimeIndex. Current tz: {data.index.tz}")
        return pd.DataFrame()

    min_bars_needed = settings.UNICORN_SWING_LOOKBACK * 4 + 10 # Rough estimate for complex patterns
    if len(data) < min_bars_needed:
        logger.warning(f"Unicorn: Not enough data ({len(data)} bars, need approx {min_bars_needed}) for robust signal generation.")
        return pd.DataFrame()

    data_with_swings = find_swing_points(data.copy(), n=settings.UNICORN_SWING_LOOKBACK)
    
    # Get potential breaker structures (sweep, MSS, breaker candle)
    # This function identifies the bar where MSS occurs and the breaker range.
    breaker_setups = find_breaker_structures(data_with_swings, lookback_for_sweep=15, breaker_candle_count=1)

    if not breaker_setups:
        logger.info("Unicorn: No initial breaker structures found.")
        return pd.DataFrame()

    # Now, for each breaker setup, look for an FVG during/after the MSS
    # and then check for entry on retracement.
    for setup in breaker_setups:
        idx_mss = setup['index_mss'] # Index of the bar that confirmed MSS
        
        # The displacement move starts roughly after the sweep and culminates in/around idx_mss.
        # We need to find an FVG formed during this displacement.
        # The FVG is formed by 3 bars. If idx_mss is the 3rd bar of displacement, FVG is on bar idx_mss-1.
        # Or if idx_mss is part of a larger displacement, FVG could be around it.
        # Search for FVG around the MSS bar (e.g., from idx_mss-3 to idx_mss)
        # find_fvg looks for FVG on bar_index+2, based on bar_index+1, bar_index+2, bar_index+3
        
        fvg_zone = None
        fvg_search_start_idx = -1

        if setup['type'] == 'bullish':
            # For bullish MSS, displacement is upwards. Look for bullish FVG.
            # The impulse move is from the low after sweep up to/beyond MSS.
            # Breaker is before this impulse. FVG is within this impulse.
            # Let's search for FVG starting from a few bars before MSS up to MSS bar.
            # If MSS is at idx_mss, the FVG could form involving (idx_mss-2, idx_mss-1, idx_mss) -> FVG on idx_mss-1
            # So, call find_fvg with bar_index = idx_mss - 3
            if idx_mss >= 3 :
                 fvg_zone = find_fvg(data_with_swings, idx_mss - 3, direction="bullish")
                 if fvg_zone: fvg_search_start_idx = idx_mss -1 # FVG is on bar idx_mss-1

        elif setup['type'] == 'bearish':
            # For bearish MSS, displacement is downwards. Look for bearish FVG.
            if idx_mss >= 3:
                fvg_zone = find_fvg(data_with_swings, idx_mss - 3, direction="bearish")
                if fvg_zone: fvg_search_start_idx = idx_mss -1


        if fvg_zone:
            fvg_low, fvg_high = fvg_zone
            breaker_low, breaker_high = setup['breaker_low'], setup['breaker_high']

            # Check for overlap between FVG and Breaker
            overlap_low = max(fvg_low, breaker_low)
            overlap_high = min(fvg_high, breaker_high)

            if overlap_low < overlap_high: # Overlap exists
                # logger.info(f"Unicorn {setup['type']} Setup Confirmed @ MSS {setup['timestamp_mss']}: "
                #             f"Breaker [{breaker_low:.2f}-{breaker_high:.2f}], "
                #             f"FVG [{fvg_low:.2f}-{fvg_high:.2f}], "
                #             f"Overlap [{overlap_low:.2f}-{overlap_high:.2f}]")

                # Now, look for entry on retracement into this overlap zone *after* its formation.
                # The FVG and breaker are confirmed around/by idx_mss (or fvg_search_start_idx).
                # We need to scan bars *after* fvg_search_start_idx for entry.
                
                # The FVG is on bar fvg_search_start_idx (e.g. idx_mss-1)
                # Entry scan starts from bar fvg_search_start_idx + 1
                for entry_idx in range(fvg_search_start_idx + 1, len(data_with_swings)):
                    entry_bar = data_with_swings.iloc[entry_idx]
                    entry_bar_time = data_with_swings.index[entry_idx]
                    
                    # Prevent re-entry on old signals if too far from formation
                    if (entry_idx - idx_mss) > settings.UNICORN_SWING_LOOKBACK * 2 : # Arbitrary limit
                        break 

                    if setup['type'] == 'bullish':
                        # Entry: Bar dips into overlap_high and closes above overlap_low (or just enters FVG)
                        if entry_bar['Low'] <= overlap_high and entry_bar['Close'] > overlap_low and entry_bar['Close'] > entry_bar['Open']:
                            entry_price = entry_bar['Close']
                            sl = entry_price - stop_loss_points # Alt: setup['breaker_low'] - buffer
                            tp = entry_price + (stop_loss_points * rrr)
                            signals_list.append({
                                'SignalTime': entry_bar_time, 'SignalType': 'Long',
                                'EntryPrice': entry_price, 'SL': sl, 'TP': tp,
                                'Reason': f"Unicorn Bullish: Retrace to Overlap Zone [{overlap_low:.2f}-{overlap_high:.2f}] formed around {setup['timestamp_mss'].time()}"
                            })
                            # logger.debug(f"Unicorn Long Signal @ {entry_bar_time}, Entry: {entry_price:.2f}")
                            break # One entry per setup instance for now
                    
                    elif setup['type'] == 'bearish':
                        if entry_bar['High'] >= overlap_low and entry_bar['Close'] < overlap_high and entry_bar['Close'] < entry_bar['Open']:
                            entry_price = entry_bar['Close']
                            sl = entry_price + stop_loss_points # Alt: setup['breaker_high'] + buffer
                            tp = entry_price - (stop_loss_points * rrr)
                            signals_list.append({
                                'SignalTime': entry_bar_time, 'SignalType': 'Short',
                                'EntryPrice': entry_price, 'SL': sl, 'TP': tp,
                                'Reason': f"Unicorn Bearish: Retrace to Overlap Zone [{overlap_low:.2f}-{overlap_high:.2f}] formed around {setup['timestamp_mss'].time()}"
                            })
                            # logger.debug(f"Unicorn Short Signal @ {entry_bar_time}, Entry: {entry_price:.2f}")
                            break # One entry per setup instance

    if not signals_list:
        logger.info("Unicorn: No signals generated with the enhanced Breaker+FVG overlap logic.")
        
    return pd.DataFrame(signals_list)

