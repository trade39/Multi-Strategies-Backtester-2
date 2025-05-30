# services/strategies/technical_utils.py
"""
Common technical analysis helper functions for strategies.
Includes identification of swing points, FVGs, and breaker block patterns.
"""
import pandas as pd
import numpy as np
from config import settings # Assuming settings.py is in a directory accessible via 'config'
from utils.logger import get_logger # Assuming logger.py is in 'utils'

logger = get_logger(__name__)

def find_fvg(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
    """
    Identifies a Fair Value Gap (FVG) based on a 3-bar pattern.
    The FVG is identified based on bars [bar_index+1, bar_index+2, bar_index+3].
    The FVG itself is the price range on bar_index+2, representing the gap.

    Args:
        data (pd.DataFrame): OHLCV data with 'High' and 'Low' columns.
        bar_index (int): The index of the bar *immediately preceding* the 3-bar FVG pattern.
                         So, the pattern involves data.iloc[bar_index+1], data.iloc[bar_index+2], data.iloc[bar_index+3].
        direction (str): "bullish" (gap below current price, formed by upward move) or
                         "bearish" (gap above current price, formed by downward move).

    Returns:
        tuple[float, float] | None: (fvg_low, fvg_high) if FVG found, else None.
                                     For Bullish FVG: (High of bar_index+1, Low of bar_index+3)
                                     For Bearish FVG: (High of bar_index+3, Low of bar_index+1)
    """
    if bar_index + 3 >= len(data): # Need at least 3 bars *after* bar_index
        return None

    bar1 = data.iloc[bar_index + 1] # First bar of the 3-bar pattern
    # bar2 = data.iloc[bar_index + 2] # Middle bar where the imbalance occurs (FVG exists on this bar's range)
    bar3 = data.iloc[bar_index + 3] # Third bar

    fvg_low, fvg_high = None, None

    if direction == "bullish":
        # Bullish FVG: Bar1's High is below Bar3's Low.
        # The FVG is the space between Bar1.High and Bar3.Low, occurring during Bar2's timeframe.
        if bar1['High'] < bar3['Low']:
            fvg_low = bar1['High']
            fvg_high = bar3['Low']
            # logger.debug(f"Bullish FVG identified at index {bar_index+2} (between {data.index[bar_index+1]} and {data.index[bar_index+3]}): Low={fvg_low:.2f}, High={fvg_high:.2f}")
            return fvg_low, fvg_high
    elif direction == "bearish":
        # Bearish FVG: Bar1's Low is above Bar3's High.
        # The FVG is the space between Bar3.High and Bar1.Low, occurring during Bar2's timeframe.
        if bar1['Low'] > bar3['High']:
            fvg_low = bar3['High']
            fvg_high = bar1['Low']
            # logger.debug(f"Bearish FVG identified at index {bar_index+2} (between {data.index[bar_index+1]} and {data.index[bar_index+3]}): Low={fvg_low:.2f}, High={fvg_high:.2f}")
            return fvg_low, fvg_high
    return None


def find_swing_points(data: pd.DataFrame, n: int = settings.UNICORN_SWING_LOOKBACK) -> pd.DataFrame:
    """
    Identifies swing highs and lows.
    A swing high is a high with 'n' lower highs on each side.
    A swing low is a low with 'n' higher lows on each side.

    Args:
        data (pd.DataFrame): OHLC data.
        n (int): Number of bars to look left and right. Sourced from settings.

    Returns:
        pd.DataFrame: DataFrame with 'SwingHigh' (price) and 'SwingLow' (price) columns, and
                      'SH_idx', 'SL_idx' (boolean indicating if it's a swing point).
    """
    data_copy = data.copy()
    data_copy['SwingHigh'] = np.nan
    data_copy['SwingLow'] = np.nan
    data_copy['SH_idx'] = False
    data_copy['SL_idx'] = False

    if n <= 0:
        logger.warning(f"Swing lookback period 'n' ({n}) must be positive. Skipping swing point calculation.")
        return data_copy
    if len(data_copy) < (2 * n + 1):
        logger.debug(f"Not enough data ({len(data_copy)} bars) for swing lookback n={n}. Returning data without swings.")
        return data_copy

    for i in range(n, len(data_copy) - n):
        is_swing_high = True
        current_high = data_copy['High'].iloc[i]
        for j in range(1, n + 1):
            if current_high < data_copy['High'].iloc[i-j] or current_high < data_copy['High'].iloc[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            data_copy.loc[data_copy.index[i], 'SwingHigh'] = current_high
            data_copy.loc[data_copy.index[i], 'SH_idx'] = True

        is_swing_low = True
        current_low = data_copy['Low'].iloc[i]
        for j in range(1, n + 1):
            if current_low > data_copy['Low'].iloc[i-j] or current_low > data_copy['Low'].iloc[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
            data_copy.loc[data_copy.index[i], 'SwingLow'] = current_low
            data_copy.loc[data_copy.index[i], 'SL_idx'] = True
            
    return data_copy

def get_last_n_candles_range(data: pd.DataFrame, current_idx: int, n_candles: int, direction: str) -> tuple[float, float] | None:
    """
    Gets the consolidated high and low of the last N candles of a specific direction
    before current_idx. Used for identifying breaker blocks.

    Args:
        data (pd.DataFrame): OHLC data.
        current_idx (int): The index *after* the block of candles.
        n_candles (int): Number of candles to look back for.
        direction (str): "up" for up-close candles, "down" for down-close candles.

    Returns:
        tuple[float, float] | None: (block_low, block_high) or None if not found.
    """
    if current_idx < n_candles:
        return None

    block_candles_data = []
    count = 0
    # Iterate backwards from current_idx - 1
    for i in range(current_idx - 1, -1, -1):
        candle = data.iloc[i]
        is_up_candle = candle['Close'] > candle['Open']
        is_down_candle = candle['Close'] < candle['Open']

        if (direction == "up" and is_up_candle) or \
           (direction == "down" and is_down_candle) or \
           (direction == "any"): # For cases where any candle type forms the block
            block_candles_data.append(candle)
            count += 1
            if count == n_candles:
                break
        elif count > 0: # If we started collecting and encounter a different type, stop
            break
            
    if not block_candles_data:
        return None

    block_df = pd.DataFrame(block_candles_data)
    return block_df['Low'].min(), block_df['High'].max()


def find_breaker_structures(data_with_swings: pd.DataFrame, lookback_for_sweep: int = 10, breaker_candle_count: int = 1):
    """
    Identifies potential breaker structures (sweep, MSS, breaker candle range).
    This is a complex function and will identify candidates. FVG overlap is checked later.

    Args:
        data_with_swings (pd.DataFrame): Data with 'High', 'Low', 'Close', 'Open', 
                                         'SwingHigh', 'SwingLow', 'SH_idx', 'SL_idx'.
        lookback_for_sweep (int): How many bars to look back for a swing point to be swept.
        breaker_candle_count (int): Number of candles to consider for the breaker block.

    Returns:
        list: A list of dictionaries, each representing a potential breaker setup.
              Each dict contains: 'index' (of MSS bar), 'type' ('bullish'/'bearish'),
              'breaker_low', 'breaker_high', 'mss_price', 'swept_liquidity_price'.
    """
    breaker_candidates = []
    if len(data_with_swings) < lookback_for_sweep + settings.UNICORN_SWING_LOOKBACK * 2 + 5: # Min data
        return []

    # Iterate through data, starting after enough bars for lookbacks
    # The current bar 'i' is where we might confirm a Market Structure Shift (MSS)
    for i in range(lookback_for_sweep + settings.UNICORN_SWING_LOOKBACK, len(data_with_swings)):
        current_bar = data_with_swings.iloc[i]

        # --- Look for Bullish Breaker Setup (MSS is an upward break of a SwingHigh) ---
        # 1. Potential MSS: Current bar's High breaks above a recent SwingHigh.
        #    Search for a relevant SwingHigh to the left of current_bar 'i'.
        for sh_idx in range(i - 1, max(i - lookback_for_sweep - 1, settings.UNICORN_SWING_LOOKBACK -1), -1):
            if data_with_swings['SH_idx'].iloc[sh_idx]:
                prev_swing_high_price = data_with_swings['SwingHigh'].iloc[sh_idx]
                prev_swing_high_time_idx = sh_idx

                if current_bar['High'] > prev_swing_high_price: # Potential MSS
                    # 2. Liquidity Sweep: Check if a SwingLow was taken out *before* this MSS
                    #    and *after or around* the time of prev_swing_high_time_idx formation.
                    #    The swept SL should ideally be lower than any SL that formed prev_swing_high.
                    for sl_idx in range(prev_swing_high_time_idx, max(prev_swing_high_time_idx - lookback_for_sweep, settings.UNICORN_SWING_LOOKBACK -1), -1):
                        if data_with_swings['SL_idx'].iloc[sl_idx]:
                            swept_swing_low_price = data_with_swings['SwingLow'].iloc[sl_idx]
                            
                            # Check for sweep: Find bars between sl_idx and current bar 'i' that went below swept_swing_low_price
                            sweep_confirmed = False
                            idx_of_sweep = -1
                            for k in range(sl_idx + 1, i): # Bars after SL formation and before MSS bar 'i'
                                if data_with_swings['Low'].iloc[k] < swept_swing_low_price:
                                    sweep_confirmed = True
                                    idx_of_sweep = k # Index of the candle that made the low (part of sweep)
                                    break
                            
                            if sweep_confirmed:
                                # 3. Identify Bullish Breaker Block: Last down-candle(s) before the rally that caused MSS.
                                #    The rally starts after idx_of_sweep and leads to bar 'i'.
                                #    We need to find the start of the rally. A simple way: candles before 'i' that are part of the up-move.
                                #    The breaker is before this rally.
                                #    Look for down candles just before the strong up-move from 'idx_of_sweep' up to 'i'.
                                #    The breaker is the last down candle(s) before the move that starts from/after the sweep low (idx_of_sweep)
                                #    and leads to the MSS at bar 'i'.
                                
                                # Search for the start of the impulsive move leading to MSS, after the sweep.
                                # The breaker would be the down-candles immediately preceding this impulse.
                                # This part is tricky. Let's simplify: find down candles before the bar 'i'.
                                # The breaker is likely between idx_of_sweep and prev_swing_high_time_idx.
                                # The most reliable breaker is often the one that leads to the sweep itself, if it's a down candle.
                                # Or, the series of down candles just before the MSS impulse.

                                # Simpler: last N down-close candles before the bar that broke structure (bar 'i').
                                # The search for breaker should be from idx_of_sweep (or slightly after) up to i-1.
                                breaker_range = None
                                for breaker_search_idx in range(i - 1, idx_of_sweep -1 , -1): # Search backwards from bar before MSS
                                    if data_with_swings['Close'].iloc[breaker_search_idx] < data_with_swings['Open'].iloc[breaker_search_idx]: # It's a down candle
                                        # Consider this down candle as the breaker (or part of it)
                                        # For simplicity, take the last single down candle before the up-move to MSS.
                                        breaker_low = data_with_swings['Low'].iloc[breaker_search_idx]
                                        breaker_high = data_with_swings['High'].iloc[breaker_search_idx]
                                        breaker_range = (breaker_low, breaker_high)
                                        # If using breaker_candle_count > 1, need more logic here.
                                        break # Found the last down candle

                                if breaker_range:
                                    breaker_candidates.append({
                                        'index_mss': i, 'timestamp_mss': data_with_swings.index[i],
                                        'type': 'bullish',
                                        'breaker_low': breaker_range[0], 'breaker_high': breaker_range[1],
                                        'mss_price': prev_swing_high_price, # The level that was broken
                                        'swept_liquidity_price': swept_swing_low_price,
                                        'idx_swept_low': sl_idx, 'idx_broken_high': prev_swing_high_time_idx
                                    })
                                    # logger.debug(f"Bullish Breaker Candidate @ {data_with_swings.index[i]}: Breaker {breaker_range}, MSS above {prev_swing_high_price}, Swept {swept_swing_low_price}")
                                    # Found one valid sequence for this MSS, can break inner loops for this 'i'
                                    break # break from sl_idx loop
                    if sweep_confirmed: break # break from sh_idx loop


        # --- Look for Bearish Breaker Setup (MSS is a downward break of a SwingLow) ---
        # 1. Potential MSS: Current bar's Low breaks below a recent SwingLow.
        for sl_idx_mss in range(i - 1, max(i - lookback_for_sweep - 1, settings.UNICORN_SWING_LOOKBACK -1), -1):
            if data_with_swings['SL_idx'].iloc[sl_idx_mss]:
                prev_swing_low_price = data_with_swings['SwingLow'].iloc[sl_idx_mss]
                prev_swing_low_time_idx = sl_idx_mss

                if current_bar['Low'] < prev_swing_low_price: # Potential MSS
                    # 2. Liquidity Sweep: Check if a SwingHigh was taken out *before* this MSS.
                    for sh_idx_sweep in range(prev_swing_low_time_idx, max(prev_swing_low_time_idx - lookback_for_sweep, settings.UNICORN_SWING_LOOKBACK-1), -1):
                        if data_with_swings['SH_idx'].iloc[sh_idx_sweep]:
                            swept_swing_high_price = data_with_swings['SwingHigh'].iloc[sh_idx_sweep]
                            
                            sweep_confirmed = False
                            idx_of_sweep = -1
                            for k in range(sh_idx_sweep + 1, i):
                                if data_with_swings['High'].iloc[k] > swept_swing_high_price:
                                    sweep_confirmed = True
                                    idx_of_sweep = k
                                    break
                            
                            if sweep_confirmed:
                                # 3. Identify Bearish Breaker Block: Last up-candle(s) before the decline.
                                breaker_range = None
                                for breaker_search_idx in range(i - 1, idx_of_sweep -1, -1):
                                     if data_with_swings['Close'].iloc[breaker_search_idx] > data_with_swings['Open'].iloc[breaker_search_idx]: # It's an up candle
                                        breaker_low = data_with_swings['Low'].iloc[breaker_search_idx]
                                        breaker_high = data_with_swings['High'].iloc[breaker_search_idx]
                                        breaker_range = (breaker_low, breaker_high)
                                        break
                                
                                if breaker_range:
                                    breaker_candidates.append({
                                        'index_mss': i, 'timestamp_mss': data_with_swings.index[i],
                                        'type': 'bearish',
                                        'breaker_low': breaker_range[0], 'breaker_high': breaker_range[1],
                                        'mss_price': prev_swing_low_price,
                                        'swept_liquidity_price': swept_swing_high_price,
                                        'idx_swept_high': sh_idx_sweep, 'idx_broken_low': prev_swing_low_time_idx
                                    })
                                    # logger.debug(f"Bearish Breaker Candidate @ {data_with_swings.index[i]}: Breaker {breaker_range}, MSS below {prev_swing_low_price}, Swept {swept_swing_high_price}")
                                    break # break from sh_idx_sweep loop
                    if sweep_confirmed: break # break from sl_idx_mss loop
                    
    return breaker_candidates

