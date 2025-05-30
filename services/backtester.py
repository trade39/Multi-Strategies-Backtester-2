# services/backtester.py
"""
Executes the backtest by simulating trades based on generated signals
and calculating profit/loss and equity curve.
"""
import pandas as pd
import numpy as np
from utils.logger import get_logger
from config.settings import DEFAULT_STRATEGY_TIMEFRAME # Corrected import

logger = get_logger(__name__)

def _parse_interval_to_timedelta(interval_str: str) -> pd.Timedelta | None:
    """
    Parses a yfinance-style interval string (e.g., "15m", "1h", "1d")
    into a pandas Timedelta object. Returns None if parsing fails.
    """
    try:
        if 'm' in interval_str and 'mo' not in interval_str:
            return pd.Timedelta(minutes=int(interval_str.replace('m', '')))
        elif 'h' in interval_str:
            return pd.Timedelta(hours=int(interval_str.replace('h', '')))
        elif 'd' in interval_str:
            return pd.Timedelta(days=int(interval_str.replace('d', '')))
        elif 'wk' in interval_str:
            return pd.Timedelta(weeks=int(interval_str.replace('wk', '')))
        logger.warning(f"Interval string '{interval_str}' not supported for Timedelta parsing for bar duration.")
        return None
    except ValueError:
        logger.error(f"Could not parse interval string '{interval_str}' to Timedelta.")
        return None

def run_backtest(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float,
    risk_per_trade_percent: float,
    stop_loss_points_config: float,
    data_interval_str: str # Explicitly require the interval string
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Runs the backtest simulation.
    Args:
        data_interval_str (str): The interval string of the price_data (e.g., "15m", "1h").
    """
    if price_data.empty:
        logger.warning("Price data is empty. Cannot run backtest.")
        return pd.DataFrame(), pd.Series(dtype=float), {}

    equity_points = {} # Using dict to store equity at specific timestamps
    # Start equity just before the first bar of price_data, or a very early date if price_data is empty (though handled)
    # This ensures the equity curve always starts from initial_capital.
    start_point_equity_time = price_data.index.min() - pd.Timedelta(microseconds=1) if not price_data.empty else pd.Timestamp('1970-01-01', tz='UTC')

    equity_points[start_point_equity_time] = initial_capital
    current_capital = initial_capital
    trades_log = []

    price_data = price_data.sort_index()

    bar_duration_info = f"Data Interval: {data_interval_str}"
    if price_data.index.freqstr:
        bar_duration_info = f"Inferred Bar Duration from Index: {price_data.index.freqstr}"
    elif len(price_data.index) > 1:
        inferred_duration = price_data.index[1] - price_data.index[0]
        bar_duration_info = f"Calculated Bar Duration from Data: {inferred_duration}"
    else: # Fallback to parsing the passed interval string
        parsed_td = _parse_interval_to_timedelta(data_interval_str)
        if parsed_td:
            bar_duration_info = f"Bar Duration (from interval string '{data_interval_str}'): {parsed_td}"
    logger.info(f"Backtesting with data. {bar_duration_info}")


    active_trade = None
    for i, (timestamp, current_bar) in enumerate(price_data.iterrows()):
        if active_trade:
            exit_reason, exit_price = None, None
            if active_trade['Type'] == 'Long':
                if current_bar['Low'] <= active_trade['SL']: exit_price, exit_reason = active_trade['SL'], 'Stop Loss'
                elif current_bar['High'] >= active_trade['TP']: exit_price, exit_reason = active_trade['TP'], 'Take Profit'
            elif active_trade['Type'] == 'Short':
                if current_bar['High'] >= active_trade['SL']: exit_price, exit_reason = active_trade['SL'], 'Stop Loss'
                elif current_bar['Low'] <= active_trade['TP']: exit_price, exit_reason = active_trade['TP'], 'Take Profit'
            
            if exit_reason:
                pnl = (exit_price - active_trade['EntryPrice']) * active_trade['PositionSize'] if active_trade['Type'] == 'Long' else \
                      (active_trade['EntryPrice'] - exit_price) * active_trade['PositionSize']
                current_capital += pnl
                active_trade.update({'ExitPrice': exit_price, 'ExitTime': timestamp, 'P&L': pnl, 'ExitReason': exit_reason})
                trades_log.append(active_trade.copy())
                equity_points[timestamp] = current_capital
                logger.debug(f"Trade closed: {active_trade['Type']} at {exit_price:.2f} ({exit_reason}). P&L: {pnl:.2f}. Capital: {current_capital:.2f}")
                active_trade = None

        if not active_trade and timestamp in signals.index:
            signal = signals.loc[timestamp]
            if isinstance(signal, pd.DataFrame): signal = signal.iloc[0]
            risk_amount = current_capital * (risk_per_trade_percent / 100.0)
            if stop_loss_points_config <= 0:
                logger.error(f"SL points ({stop_loss_points_config}) non-positive. Skipping trade at {timestamp}.")
            else:
                position_size = risk_amount / stop_loss_points_config
                active_trade = {'EntryTime': timestamp, 'EntryPrice': signal['EntryPrice'], 'Type': signal['SignalType'],
                                'SL': signal['SL'], 'TP': signal['TP'], 'PositionSize': position_size,
                                'InitialRiskAmount': risk_amount, 'StopLossPoints': stop_loss_points_config}
                logger.debug(f"Trade opened: {active_trade['Type']} at {active_trade['EntryPrice']:.2f} on {timestamp}. PosSize: {position_size:.2f}.")
    
    if active_trade:
        last_close = price_data['Close'].iloc[-1]
        pnl = (last_close - active_trade['EntryPrice']) * active_trade['PositionSize'] if active_trade['Type'] == 'Long' else \
              (active_trade['EntryPrice'] - last_close) * active_trade['PositionSize']
        current_capital += pnl
        active_trade.update({'ExitPrice': last_close, 'ExitTime': price_data.index[-1], 'P&L': pnl, 'ExitReason': 'End of Data'})
        trades_log.append(active_trade.copy())
        equity_points[price_data.index[-1]] = current_capital
        logger.debug(f"Trade closed at EOD: {active_trade['Type']} at {last_close:.2f}. P&L: {pnl:.2f}. Capital: {current_capital:.2f}")

    trades_df = pd.DataFrame(trades_log)
    if not trades_df.empty:
        trades_df['EntryTime'] = pd.to_datetime(trades_df['EntryTime'])
        trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime'])

    # Construct final equity series
    if not price_data.empty:
        # Ensure the final capital point is included if no trades occurred or last trade closed before last bar
        if price_data.index[-1] not in equity_points:
            equity_points[price_data.index[-1]] = current_capital
        
        equity_series_raw = pd.Series(equity_points).sort_index()
        # Reindex to the full price_data index and forward-fill
        equity_series = equity_series_raw.reindex(price_data.index, method='ffill')
        # If the first point of price_data is before the first equity point (e.g. initial capital point)
        # or if equity_series is all NaN (no trades, price_data exists), fill with initial_capital
        if equity_series.empty or pd.isna(equity_series.iloc[0]):
             # Create a series starting with initial capital at the beginning of price_data
             temp_equity = pd.Series(index=price_data.index, dtype=float)
             temp_equity.iloc[0] = initial_capital
             equity_series = temp_equity.ffill() # Fill initial capital forward
             # Then merge/update with actual trade equity points if any
             if not equity_series_raw.empty:
                 equity_series.update(equity_series_raw) # Update with actual trade points
                 equity_series = equity_series.ffill()    # Fill forward again after update

        # Ensure the very first value is initial_capital if it got lost
        if not equity_series.empty and equity_series.index.min() == price_data.index.min() and pd.isna(equity_series.iloc[0]):
            equity_series.iloc[0] = initial_capital
            equity_series = equity_series.ffill()


    else: # No price data
        equity_series = pd.Series(dtype=float)
    
    equity_series.name = "Equity"


    performance = {'Final Capital': current_capital, 'Total Trades': 0, 'Total P&L': 0, 'Win Rate': 0, 'Profit Factor': 0, 'Max Drawdown (%)': 0}
    if not trades_df.empty:
        performance['Total Trades'] = len(trades_df)
        performance['Total P&L'] = trades_df['P&L'].sum()
        gross_profit = trades_df[trades_df['P&L'] > 0]['P&L'].sum()
        gross_loss = trades_df[trades_df['P&L'] < 0]['P&L'].sum()
        performance['Winning Trades'] = len(trades_df[trades_df['P&L'] > 0])
        performance['Losing Trades'] = len(trades_df[trades_df['P&L'] < 0])
        performance['Win Rate'] = (performance['Winning Trades'] / performance['Total Trades'] * 100) if performance['Total Trades'] > 0 else 0
        performance['Profit Factor'] = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf if gross_profit > 0 else 0
        performance['Average Trade P&L'] = trades_df['P&L'].mean()
        performance['Average Winning Trade'] = trades_df[trades_df['P&L'] > 0]['P&L'].mean() if performance['Winning Trades'] > 0 else 0
        performance['Average Losing Trade'] = trades_df[trades_df['P&L'] < 0]['P&L'].mean() if performance['Losing Trades'] > 0 else 0
        
        if not equity_series.empty and equity_series.notna().any(): # Check if series has non-NaN values
            cumulative_max = equity_series.cummax()
            drawdown = (equity_series - cumulative_max) / cumulative_max
            performance['Max Drawdown (%)'] = drawdown.min() * 100 if drawdown.notna().any() and not drawdown.empty else 0
        else: performance['Max Drawdown (%)'] = 0
    
    logger.info(f"Backtest complete. Final Capital: {current_capital:.2f}. Total P&L: {performance.get('Total P&L', 0):.2f}")
    return trades_df, equity_series, performance
