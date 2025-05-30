# utils/plotting.py
"""
Functions for creating visualizations using Plotly.
Handles duplicate entries in optimization results for heatmap generation.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from config.settings import PLOTLY_TEMPLATE
from utils.logger import get_logger

# Instantiate logger
logger = get_logger(__name__)

def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """
    Plots the equity curve.

    Args:
        equity_curve (pd.Series): Series containing equity values over time.
        title (str): Title of the plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()
    if not equity_curve.empty:
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Equity", template=PLOTLY_TEMPLATE, height=500, hovermode="x unified")
    return fig

def plot_wfo_equity_curve(
    chained_oos_equity: pd.Series,
    title: str = "Walk-Forward Out-of-Sample Equity Curve"
) -> go.Figure:
    """
    Plots the chained out-of-sample equity curve from Walk-Forward Optimization.

    Args:
        chained_oos_equity (pd.Series): Series containing chained OOS equity values.
        title (str): Title of the plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()
    if not chained_oos_equity.empty:
        fig.add_trace(go.Scatter(x=chained_oos_equity.index, y=chained_oos_equity, mode='lines', name='WFO OOS Equity'))
    else:
        # Display a message if no data
        fig.add_annotation(text="No out-of-sample equity data to display.", showarrow=False, yshift=10)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        template=PLOTLY_TEMPLATE,
        height=500,
        hovermode="x unified"
    )
    return fig


def plot_trades_on_price(price_data: pd.DataFrame, trades: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Plots trades (entry and exit points) overlaid on the price candlestick chart.

    Args:
        price_data (pd.DataFrame): OHLC price data.
        trades (pd.DataFrame): DataFrame of executed trades.
        symbol (str): The financial symbol being plotted.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    if not price_data.empty:
        fig.add_trace(go.Candlestick(x=price_data.index,
                                    open=price_data['Open'],
                                    high=price_data['High'],
                                    low=price_data['Low'],
                                    close=price_data['Close'],
                                    name=f'{symbol} Price'), row=1, col=1)

    if not trades.empty:
        long_entries = trades[trades['Type'] == 'Long']
        fig.add_trace(go.Scatter(x=long_entries['EntryTime'], y=long_entries['EntryPrice'],
                                 mode='markers', name='Long Entry',
                                 marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
        
        short_entries = trades[trades['Type'] == 'Short']
        fig.add_trace(go.Scatter(x=short_entries['EntryTime'], y=short_entries['EntryPrice'],
                                 mode='markers', name='Short Entry',
                                 marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

        fig.add_trace(go.Scatter(x=trades['ExitTime'], y=trades['ExitPrice'],
                                 mode='markers', name='Exit',
                                 marker=dict(color='blue', size=8, symbol='square')), row=1, col=1)
        
        # Optional: SL/TP lines (can be noisy)
        # for _, trade in trades.iterrows():
        #     fig.add_shape(type="line", x0=trade['EntryTime'], y0=trade['SL'], x1=trade['ExitTime'], y1=trade['SL'], line=dict(color="rgba(255,0,0,0.3)", width=1, dash="dash"))
        #     fig.add_shape(type="line", x0=trade['EntryTime'], y0=trade['TP'], x1=trade['ExitTime'], y1=trade['TP'], line=dict(color="rgba(0,255,0,0.3)", width=1, dash="dash"))

    fig.update_layout(title=f'Trades for {symbol}', xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, template=PLOTLY_TEMPLATE, height=600, hovermode="x unified")
    return fig


def plot_optimization_heatmap(
    optimization_results_df: pd.DataFrame,
    param1_name: str,  # Typically the column for x-axis (e.g., 'SL Points')
    param2_name: str,  # Typically the index for y-axis (e.g., 'RRR')
    metric_name: str   # The value to be shown in the heatmap cells
) -> go.Figure:
    """
    Generates a heatmap for visualizing optimization results between two parameters.
    Handles duplicate index/column entries by averaging the metric.

    Args:
        optimization_results_df (pd.DataFrame): DataFrame of optimization results.
        param1_name (str): Name of the first parameter (columns).
        param2_name (str): Name of the second parameter (rows).
        metric_name (str): Name of the metric to plot as heatmap values.

    Returns:
        go.Figure: Plotly figure object.
    """
    if optimization_results_df.empty or not all(p in optimization_results_df.columns for p in [param1_name, param2_name, metric_name]):
        logger.warning(f"Insufficient data or missing columns for heatmap. Params: {param1_name}, {param2_name}. Metric: {metric_name}. Columns available: {optimization_results_df.columns.tolist()}")
        fig = go.Figure()
        fig.update_layout(title=f"Insufficient Data for Heatmap ({metric_name})", height=400, template=PLOTLY_TEMPLATE)
        fig.add_annotation(text="Not enough data or missing columns for heatmap generation.", showarrow=False)
        return fig

    try:
        # Ensure param1_name and param2_name exist
        if param1_name not in optimization_results_df.columns:
            raise ValueError(f"Parameter '{param1_name}' not found in optimization results columns: {optimization_results_df.columns.tolist()}")
        if param2_name not in optimization_results_df.columns:
            raise ValueError(f"Parameter '{param2_name}' not found in optimization results columns: {optimization_results_df.columns.tolist()}")
        if metric_name not in optimization_results_df.columns:
             raise ValueError(f"Metric '{metric_name}' not found in optimization results columns: {optimization_results_df.columns.tolist()}")


        # Handle potential duplicate (param1_name, param2_name) combinations by averaging the metric
        # This is crucial for the pivot operation to succeed.
        # We group by the two parameters that will form the axes of the heatmap
        # and then select the metric_name, taking the mean.
        # .reset_index() is called to turn the grouped Series back into a DataFrame suitable for pivot.
        
        logger.debug(f"Original optimization_results_df for heatmap (head):\n{optimization_results_df[[param1_name, param2_name, metric_name]].head()}")

        # Check for NaN values in grouping columns before aggregation
        if optimization_results_df[[param1_name, param2_name]].isnull().any().any():
            logger.warning(f"NaN values found in grouping columns ('{param1_name}', '{param2_name}') for heatmap. Dropping rows with NaNs in these columns.")
            cleaned_df = optimization_results_df.dropna(subset=[param1_name, param2_name])
        else:
            cleaned_df = optimization_results_df
        
        if cleaned_df.empty:
            logger.warning("DataFrame became empty after attempting to clean NaNs from grouping columns for heatmap.")
            fig = go.Figure()
            fig.update_layout(title=f"No Valid Data for Heatmap ({metric_name}) after NaN cleaning", height=400, template=PLOTLY_TEMPLATE)
            fig.add_annotation(text="Data became empty after cleaning NaNs from axis parameters.", showarrow=False)
            return fig

        # Aggregate duplicate entries
        aggregated_df = cleaned_df.groupby([param2_name, param1_name], as_index=False)[metric_name].mean()
        logger.debug(f"Aggregated_df for heatmap (head):\n{aggregated_df.head()}")

        # Pivot the aggregated data to create a matrix for the heatmap
        heatmap_data = aggregated_df.pivot(index=param2_name, columns=param1_name, values=metric_name)
        
        # Sort index (param2_name, typically RRR) in descending order for conventional heatmap display
        heatmap_data = heatmap_data.sort_index(ascending=False) 
        logger.debug(f"Pivoted heatmap_data (head):\n{heatmap_data.head()}")
        
        fig = px.imshow(heatmap_data, 
                        labels=dict(x=param1_name, y=param2_name, color=metric_name),
                        x=heatmap_data.columns, 
                        y=heatmap_data.index, 
                        aspect="auto",
                        color_continuous_scale=px.colors.diverging.RdYlGn if "P&L" in metric_name or "Ratio" in metric_name or "Factor" in metric_name else px.colors.sequential.Viridis,
                        origin='lower' # Ensures y-axis starts from bottom with sorted data
                       )
        
        fig.update_layout(title=f'Optimization Heatmap: {metric_name} vs. {param1_name} & {param2_name}',
                          xaxis_title=param1_name, yaxis_title=param2_name, height=600, template=PLOTLY_TEMPLATE)
        
        # Ensure x and y axes ticks are formatted correctly, especially for float values
        fig.update_xaxes(type='category', tickvals=heatmap_data.columns, ticktext=[f"{x:.2f}" if isinstance(x, float) else str(x) for x in heatmap_data.columns])
        fig.update_yaxes(type='category', tickvals=heatmap_data.index, ticktext=[f"{y:.1f}" if isinstance(y, float) else str(y) for y in heatmap_data.index])

    except Exception as e:
        logger.error(f"Error creating heatmap for metric '{metric_name}' with params '{param1_name}', '{param2_name}': {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(title=f"Error Generating Heatmap: Review Logs", height=400, template=PLOTLY_TEMPLATE)
        fig.add_annotation(text=f"Could not generate heatmap. Details: {str(e)}", showarrow=False)
    return fig
