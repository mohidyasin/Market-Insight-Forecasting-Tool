import pandas as pd
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from pandas.tseries.offsets import BDay
import streamlit as st
from tvDatafeed import TvDatafeed, Interval
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Market Forecast Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to filter data to a specific date
def filter_data_to_date(data, cutoff_date):
    """
    Filter data to only include dates up to and including the cutoff date.
    """
    # Convert cutoff_date to datetime if it's a string
    if isinstance(cutoff_date, str):
        cutoff_date = pd.to_datetime(cutoff_date)
    
    # Ensure cutoff_date is a Timestamp for consistent comparison
    if not isinstance(cutoff_date, pd.Timestamp):
        cutoff_date = pd.Timestamp(cutoff_date)
    
    # Convert data index to Timestamps if they aren't already
    if len(data.index) > 0:
        if not isinstance(data.index[0], pd.Timestamp):
            data.index = pd.to_datetime(data.index)
    
    # Create a mask for dates up to and including the cutoff date
    mask = data.index <= cutoff_date
    
    # Return the filtered dataframe
    return data[mask]

# Function to find the closest available date in data
def find_closest_date(data, target_date):
    """
    Find the closest available date in the data to the target date.
    """
    # Convert target_date to Timestamp if it's a string or date
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    elif hasattr(target_date, 'date'):  # datetime.date object
        target_dt = pd.Timestamp(target_date)
    else:
        target_dt = pd.Timestamp(target_date)
    
    # Ensure data index is in Timestamp format
    if len(data.index) > 0 and not isinstance(data.index[0], pd.Timestamp):
        data_dates = pd.to_datetime(data.index)
    else:
        data_dates = data.index
    
    # Find dates that are <= target_date (no looking into future)
    valid_dates = data_dates[data_dates <= target_dt]
    
    if len(valid_dates) == 0:
        return None
    
    # Return the most recent valid date as a date object for consistency
    closest_date = valid_dates.max()
    return closest_date.date()

# Calculate future business dates
def get_business_dates(start_date, num_days):
    """Calculate future business dates from a starting date."""
    dates = []
    date = pd.to_datetime(start_date)
    while len(dates) < num_days:
        date = date + timedelta(days=1)
        # Check if it's a weekday (0-4 represent Monday-Friday)
        if date.weekday() < 5:
            dates.append(date)
    return dates

# Slope Function
def calculate_slope(series, window):
    slopes = [0] * window
    for i in range(window, len(series)):
        y = series[i-window:i]
        x = list(range(window))
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)
    return slopes

# Function to compute the rolling standardization of a time series up to a specific date
def rolling_standardize_up_to_date(series, window, end_date):
    # Calculate the date to start excluding data (30 days before the target date)
    exclusion_start_date = end_date - pd.Timedelta(days=30)
    
    # Calculate rolling mean and std up to the exclusion start date
    rolling_mean = series[:exclusion_start_date].rolling(window=window, min_periods=90).mean()
    rolling_std = series[:exclusion_start_date].rolling(window=window, min_periods=90).std(ddof=0)

    # Forward-fill the rolling mean and std to extend to the end date
    extended_rolling_mean = rolling_mean.reindex(series.index, method='ffill')
    extended_rolling_std = rolling_std.reindex(series.index, method='ffill')

    # Standardize the series up to the end date using these extended rolling statistics
    standardized_series = (series - extended_rolling_mean) / extended_rolling_std.replace(0, 1)
    return standardized_series.loc[:end_date]

# Function to standardize data up to a given date
def standardize_data_up_to_date(data, features, window_size, target_date_str):
    # Convert the target date string to a datetime object
    target_date = pd.to_datetime(target_date_str)
    # Create a copy of the data to avoid modifying the original DataFrame
    temp_data = data.copy()

    # Apply the rolling standardization to each feature
    for feature in features:
        temp_data[feature] = rolling_standardize_up_to_date(temp_data[feature], window_size, target_date).round(2)

    # Drop rows with null values that may have been created due to rolling operations
    temp_data.dropna(inplace=True)
    return temp_data

# Similarity Measure
def lorentzian_distance(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return np.sum(np.log1p(np.abs(x - y)))

# Similar Dates
def get_similar_dates_by_dist(data, target_date_str, dist_func, features, top_n=20):
    # Convert the target date string to a datetime object
    target_date = pd.to_datetime(target_date_str)
    # Ensure the DataFrame index is sorted
    data = data.sort_index()
    # Find the position of the target date to prevent looking ahead
    target_index = data.index.get_loc(target_date)
    # Initialize the list for storing distances
    distances = []
    # Iterate over the DataFrame up to the target date
    for idx, row in data[features].iloc[:target_index].iterrows():
        # Calculate the distance and append it with the date as a key
        dist = dist_func(row.values, data.loc[target_date, features].values)
        distances.append((idx, dist))
    # Sort the distances by the distance values (smallest distance first)
    distances.sort(key=lambda x: x[1])
    # Extract the dates and their corresponding distances from the first 'top_n' entries
    similar_dates_with_scores = [(date.strftime('%Y-%m-%d'), dist) for date, dist in distances[:top_n]]

    return similar_dates_with_scores

# Forward returns - modified to use dynamic period
def get_forward_returns(dates, df, target_date, forward_period):
    forward_returns = []
    period_column = f'{forward_period}d_Forward_Return'
    
    for date in dates:
        if date in df.index:
            idx = df.index.get_loc(date)
            # Ensure there's enough data to calculate forward return
            forward_return = df.loc[date, period_column] if idx < len(df) - (forward_period - 1) else None
            forward_returns.append((date, forward_return))
    return forward_returns

# Generate samples - modified to use dynamic column name
def generate_samples(forward_return_period, data_feats_fwd_ret, num_samples=1000):
    column_name = f'{forward_return_period}D_Forward_Return'
    forward_returns = data_feats_fwd_ret[column_name].dropna().astype(float)
    
    if len(forward_returns) == 0:
        logger.warning(f"No valid forward returns found for {forward_return_period}D period")
        return np.array([])
    
    density = gaussian_kde(forward_returns)
    min_data, max_data = forward_returns.min(), forward_returns.max()
    x = np.linspace(min_data, max_data, 100)  # 100 points for KDE evaluation

    def inverse_transform_sampling(density_func, min_data, max_data, num_samples):
        x = np.linspace(min_data, max_data, 1000)
        cumulative_density = np.cumsum(density_func(x))
        cumulative_density /= cumulative_density[-1]  # Normalize
        samples = np.random.rand(num_samples)
        return np.interp(samples, cumulative_density, x)

    resampled_values = density(x)
    samples = inverse_transform_sampling(density, min_data, max_data, num_samples)
    return samples

# Function to convert return percentages to price
def return_to_price(ret, close_price):
    return close_price * (1 + ret / 100)

# Plot samples with color-coded ranges - modified to use dynamic period
def plot_samples(samples, forward_return_period, mean_resampled, median_resampled, target_date_str, close_price_on_target_date, ticker, next_day_str, ax):
    # Define return ranges for color coding
    ranges = [
        (-float('inf'), -3, 'darkred', '< -3%'),
        (-3, -2, 'red', '-3% to -2%'),
        (-2, -1, 'lightcoral', '-2% to -1%'),
        (-1, 0, 'mistyrose', '-1% to 0%'),
        (0, 1, 'lightgreen', '0% to +1%'),
        (1, 2, 'limegreen', '+1% to +2%'),
        (2, 3, 'green', '+2% to +3%'),
        (3, float('inf'), 'darkgreen', '> +3%')
    ]
    
    # Create histogram with color-coded bars
    n_bins = 50
    counts, bins, patches = ax.hist(samples, bins=n_bins, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color each bar based on its range and calculate range probabilities
    range_probabilities = {}
    range_centers = {}
    range_max_heights = {}
    
    # Initialize tracking for each range
    for range_min, range_max, color, label in ranges:
        range_probabilities[label] = 0
        range_centers[label] = []
        range_max_heights[label] = 0
    
    # Color bars and calculate probabilities
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        bin_width = bins[i+1] - bins[i]
        bin_probability = counts[i] * bin_width  # Convert density to probability
        
        # Find which range this bin belongs to
        for range_min, range_max, color, label in ranges:
            if range_min <= bin_center < range_max:
                patch.set_facecolor(color)
                range_probabilities[label] += bin_probability
                range_centers[label].append(bin_center)
                range_max_heights[label] = max(range_max_heights[label], counts[i])
                break
    
    # Add probability labels inside each colored section
    for range_min, range_max, color, label in ranges:
        prob_percent = range_probabilities[label] * 100
        
        # Only add label if probability is significant (>1%)
        if prob_percent > 1.0 and range_centers[label]:
            # Calculate the center position for the label
            center_x = np.mean(range_centers[label])
            center_y = range_max_heights[label] * 0.6  # Position at 60% of max height
            
            # Determine text color for visibility
            text_color = 'white' if color in ['darkred', 'red', 'green', 'darkgreen'] else 'black'
            
            # Add the probability label
            ax.text(center_x, center_y, f'{prob_percent:.1f}%', 
                   ha='center', va='center', fontweight='bold', 
                   fontsize=9, color=text_color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='black'))
    
    # Calculate probabilities for each symmetric range for legend
    prob_1 = np.sum((samples >= -1) & (samples <= 1)) / len(samples) * 100
    prob_1_2 = np.sum(((samples >= 1) & (samples <= 2)) | ((samples >= -2) & (samples <= -1))) / len(samples) * 100
    prob_2_3 = np.sum(((samples >= 2) & (samples <= 3)) | ((samples >= -3) & (samples <= -2))) / len(samples) * 100
    prob_beyond_3 = np.sum((samples > 3) | (samples < -3)) / len(samples) * 100
    
    range_probs = [
        (f'±1%: {prob_1:.1f}%', 'lightblue'),
        (f'±1% to ±2%: {prob_1_2:.1f}%', 'orange'),
        (f'±2% to ±3%: {prob_2_3:.1f}%', 'purple'),
        (f'Beyond ±3%: {prob_beyond_3:.1f}%', 'brown')
    ]
    
    # Compute and plot KDE
    density = gaussian_kde(samples)
    min_data, max_data = np.min(samples), np.max(samples)
    x = np.linspace(min_data, max_data, 200)
    ax.plot(x, density(x), color='black', linewidth=2, alpha=0.8)
    
    # Set labels and formatting
    ax.set_xlabel('Return (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    # Plot mean and median lines
    ax.axvline(x=mean_resampled, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_resampled:.2f}%')
    ax.axvline(x=median_resampled, color='blue', linestyle='--', linewidth=2, 
               label=f'Median: {median_resampled:.2f}%')
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add a secondary x-axis to show corresponding prices
    ax2 = ax.secondary_xaxis('top', functions=(lambda r: return_to_price(r, close_price_on_target_date),
                                               lambda p: (p / close_price_on_target_date - 1) * 100))
    ax2.set_xlabel(f'Price Target (Based on {target_date_str} Close: ${close_price_on_target_date:.2f})', 
                   color='darkblue', fontsize=10)
    ax2.tick_params(axis='x', labelcolor='darkblue')
    
    # Create custom legend combining mean/median with probability ranges
    legend_elements = []
    
    # Add mean/median to legend
    from matplotlib.lines import Line2D
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_resampled:.2f}%'))
    legend_elements.append(Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label=f'Median: {median_resampled:.2f}%'))
    
    # Add probability ranges to legend
    from matplotlib.patches import Patch
    for prob_text, color in range_probs:
        legend_elements.append(Patch(facecolor=color, alpha=0.7, label=prob_text))
    
    # Position legend
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=9)
    
    # Set title
    ax.set_title(f'{ticker} {forward_return_period}-Day Forward Returns Distribution\nAnalysis Date: {target_date_str} → Forecast Period: {next_day_str}', 
                fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add text box with key statistics
    stats_text = f'Samples: {len(samples):,}\nUpside Prob (>0%): {np.sum(samples > 0)/len(samples)*100:.1f}%\nDownside Prob (<0%): {np.sum(samples < 0)/len(samples)*100:.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Custom implementation of RSI
def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=length-1, adjust=False).mean()
    ema_down = down.ewm(com=length-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

# Custom implementation of EMA
def ema(series, length=20):
    return series.ewm(span=length, adjust=False).mean()

def main():
    st.title("Market Forecast Tool")
    
    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    
    # Ticker options
    ticker_options = [
        {'symbol': 'SPY', 'exchange': 'AMEX', 'name': 'S&P 500 ETF'},
        {'symbol': 'QQQ', 'exchange': 'NASDAQ', 'name': 'NASDAQ 100 ETF'},
        {'symbol': 'AAPL', 'exchange': 'NASDAQ', 'name': 'Apple'},
        {'symbol': 'MSFT', 'exchange': 'NASDAQ', 'name': 'Microsoft'},
        {'symbol': 'GOOGL', 'exchange': 'NASDAQ', 'name': 'Google'},
        {'symbol': 'AMZN', 'exchange': 'NASDAQ', 'name': 'Amazon'},
        {'symbol': 'NVDA', 'exchange': 'NASDAQ', 'name': 'Nvidia'},
        {'symbol': 'TSLA', 'exchange': 'NASDAQ', 'name': 'Tesla'}
    ]
    
    selected_name = st.sidebar.selectbox("Select Ticker:", [item['name'] for item in ticker_options], index=0)
    selected_ticker = next((item for item in ticker_options if item['name'] == selected_name), None)
    
    # Forward return period selection
    forward_periods = [1, 5, 10, 20]
    selected_period = st.sidebar.selectbox("Select Forward Return Period (days):", forward_periods, index=1)
    
    # Date selection - default to latest system date
    max_date = datetime.now().date()  # Today's date
    min_date = max_date - timedelta(days=365*3)    # 3 years ago
    
    target_date = st.sidebar.date_input(
        "Select Analysis Date:",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    # TVDatafeed credentials
    username = 'gme_bagholder1'
    password = 'Mohidabdullah9'
    
    # Analyze button
    if st.sidebar.button("Generate Forecast"):
        with st.spinner("Retrieving data and generating forecast..."):
            symbol = selected_ticker['symbol']
            exchange = selected_ticker['exchange']
            
            # Initialize TVDatafeed
            try:
                tv = TvDatafeed(username, password)
                
                # Fetch data
                data = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=10000)
                
                if data is None or len(data) == 0:
                    st.error(f"Failed to retrieve data for {symbol} on {exchange}")
                    return
                
                # Reset the index to make datetime a column
                data = data.reset_index()
                
                # Rename columns for consistency
                data = data[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(
                    columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
                )
                
                # Convert Date to datetime and extract date part only
                data['Date'] = pd.to_datetime(data['Date'])
                data['Date'] = data['Date'].dt.date
                
                # Set Date as index
                data.set_index('Date', inplace=True)
                
                # Check if target date exists in data, if not find closest available date
                original_target_date = target_date_str
                closest_date = find_closest_date(data, target_date)
                
                if closest_date is None:
                    st.error(f"No data available for or before the selected date {target_date_str}")
                    logger.error(f"No data available for or before date {target_date_str}")
                    return
                
                if closest_date != target_date:
                    target_date_str = closest_date.strftime('%Y-%m-%d')
                    st.warning(f"Selected date {original_target_date} not found in data. Using closest available date: {target_date_str}")
                    logger.info(f"Date {original_target_date} not found, using closest date {target_date_str}")
                
                # Filter data to only include dates up to the target date
                data = filter_data_to_date(data, target_date_str)
                
                st.success(f"Retrieved {len(data)} days of historical data up to {target_date_str}")
                
                # Calculate technical indicators
                data['RSI_14'] = rsi(data['Close'], length=14)
                data['5d_avg_RSI'] = data['RSI_14'].rolling(window=5).mean().round(2)
                data['RSI_14_pct_change'] = data['RSI_14'].pct_change().mul(100).round(2)
                data['RSI_14_Slope'] = calculate_slope(data['RSI_14'], 3)
                data['RSI_14_Slope'] = data['RSI_14_Slope'].round(2)
                data['20d_EMA'] = ema(data['Close'], length=20).round(2)
                data['Pct_Distance_to_EMA'] = ((data['Close'] - data['20d_EMA']) / data['20d_EMA']).mul(100).round(2)
                data['EMA_Slope'] = calculate_slope(data['20d_EMA'], 3)
                data['EMA_Slope'] = data['EMA_Slope'].round(2)
                
                # Calculate ATR
                high_low = data['High'] - data['Low']
                high_close = abs(data['High'] - data['Close'].shift())
                low_close = abs(data['Low'] - data['Close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data['ATR_14'] = true_range.ewm(span=14, adjust=False).mean().round(2)
                data['ATR_14_rank'] = data['ATR_14'].rolling(window=252).apply(lambda x: x.rank(pct=True)[-1]).round(2)
                
                # Volume indicators
                data['Volume_avg_5d_ema'] = ema(data['Volume'], length=5).round(2)
                data['Volume_ema_slope'] = calculate_slope(data['Volume_avg_5d_ema'], 3)
                data['Volume_ema_slope'] = data['Volume_ema_slope'].round(2)
                data['Volume_avg_5d_ema_rank'] = data['Volume_avg_5d_ema'].rolling(window=252).apply(lambda x: x.rank(pct=True)[-1]).round(2)
                
                # Convert index to datetime and drop NaN values
                data.index = pd.to_datetime(data.index)
                data.dropna(inplace=True)
                
                # Features to standardize
                excluded_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                features = data.columns.difference(excluded_columns)
                window_size = 252
                standardized_data = standardize_data_up_to_date(data, features, window_size, target_date_str)
                
                # Find similar dates
                lorentzian_dates = get_similar_dates_by_dist(standardized_data, target_date_str, lorentzian_distance, features)
                
                # Calculate returns - now dynamic based on selected period
                standardized_data['Returns'] = (standardized_data['Close'].pct_change() * 100).round(2)
                standardized_data['1d_Forward_Return'] = standardized_data['Returns'].shift(-1)
                standardized_data['5d_Forward_Return'] = (((standardized_data['Close'].shift(-4) / standardized_data['Close']) - 1) * 100).round(2)
                standardized_data['10d_Forward_Return'] = (((standardized_data['Close'].shift(-9) / standardized_data['Close']) - 1) * 100).round(2)
                standardized_data['20d_Forward_Return'] = (((standardized_data['Close'].shift(-19) / standardized_data['Close']) - 1) * 100).round(2)
                
                # Get forward returns for similar dates
                lorentzian_dates_only = [date for date, _ in lorentzian_dates]
                forward_returns_list = get_forward_returns(lorentzian_dates_only, standardized_data, target_date_str, selected_period)
                sorted_forward_returns = sorted(forward_returns_list, key=lambda x: x[0])
                data_feats_fwd_ret = pd.DataFrame(sorted_forward_returns, columns=['Date', f'{selected_period}D_Forward_Return'])
                
                # Generate samples
                np.random.seed(123)
                samples = generate_samples(selected_period, data_feats_fwd_ret)
                
                if len(samples) == 0:
                    st.error(f"Could not generate samples for {selected_period}-day forward returns. Not enough similar historical patterns found.")
                    return
                
                # Calculate next trading day and forecast end date
                next_day = pd.to_datetime(target_date_str) + BDay(1)
                next_day_str = next_day.strftime('%Y-%m-%d')
                
                # Calculate the end date (selected_period business days from target date)
                business_days = get_business_dates(target_date_str, selected_period)
                end_date = business_days[-1].strftime('%Y-%m-%d')
                
                # Get close price on target date
                close_price_on_target_date = standardized_data.loc[target_date_str, 'Close']
                
                # Display enhanced forecast summary
                st.subheader(f"{selected_period}-Day Forecast ({next_day_str} to {end_date})")
                
                # Calculate today's return if available
                today_return = "N/A"
                try:
                    target_idx = standardized_data.index.get_loc(pd.to_datetime(target_date_str))
                    if target_idx > 0:
                        today_return = standardized_data['Returns'].iloc[target_idx]
                except:
                    pass
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Today's Return", f"{today_return}%" if isinstance(today_return, (int, float)) else today_return)
                col2.metric("Mean Expected Return", f"{np.mean(samples):.2f}%")
                col3.metric("Median Expected Return", f"{np.median(samples):.2f}%")
                
                # Create forecast plot
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_samples(samples, selected_period, np.mean(samples), np.median(samples), 
                             target_date_str, close_price_on_target_date, symbol, next_day_str, ax)
                st.pyplot(fig)
                
                # Show historical price paths
                st.subheader(f"Historical Price Paths After Similar Dates ({selected_period} Days)")
                
                # Calculate future dates for projection (selected_period business days)
                future_dates = pd.date_range(start=pd.to_datetime(target_date_str), periods=selected_period+1, freq='B')
                
                # Create a figure for the historical paths
                fig_paths, ax_paths = plt.subplots(figsize=(12, 7))
                
                # Get recent price data for context
                recent_days = min(90, len(data))
                chart_data = data.iloc[-recent_days:].copy()
                
                # Plot the current price history
                ax_paths.plot(chart_data.index, chart_data['Close'], label='Current Price', color='blue', linewidth=2)
                
                # Highlight the target date
                target_date_idx = chart_data.index.get_loc(pd.to_datetime(target_date_str)) if pd.to_datetime(target_date_str) in chart_data.index else -1
                if target_date_idx >= 0:
                    ax_paths.scatter(chart_data.index[target_date_idx], chart_data['Close'].iloc[target_date_idx], 
                                    color='red', s=100, zorder=5)
                    ax_paths.axvline(x=chart_data.index[target_date_idx], color='red', linestyle='--', alpha=0.5)
                
                # Plot future paths from similar dates
                colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Use a colormap for different dates
                legend_items = []
                
                # Keep track of original data for paths
                original_data = data.copy()
                
                for i, (date_str, _) in enumerate(lorentzian_dates[:10]):  # Take top 10 similar dates
                    date = pd.to_datetime(date_str)
                    
                    # Find the index position in the original data
                    if date in original_data.index:
                        idx = original_data.index.get_loc(date)
                        
                        # Make sure we have at least selected_period more days of data after this date
                        if idx + selected_period < len(original_data):
                            # Extract the future path
                            future_dates_hist = original_data.index[idx:idx+selected_period+1]
                            future_prices = original_data['Close'].iloc[idx:idx+selected_period+1].values
                            
                            # Normalize the path to start at the same point as the target date close price
                            price_ratio = chart_data.iloc[target_date_idx]['Close'] / future_prices[0]
                            normalized_prices = future_prices * price_ratio
                            
                            # Plot the future path
                            path_dates = pd.date_range(start=chart_data.index[target_date_idx], periods=selected_period+1, freq='B')
                            line, = ax_paths.plot(path_dates, 
                                                normalized_prices,
                                                label=f"Path after {date_str}", 
                                                color=colors[i % len(colors)],
                                                alpha=0.7, 
                                                linewidth=1.5,
                                                linestyle='-')
                            
                            # Add a marker for the end point
                            ax_paths.scatter(path_dates[-1], normalized_prices[-1], color=colors[i % len(colors)], s=30)
                            
                            # Calculate the percent change for this path
                            pct_change = ((normalized_prices[-1] / normalized_prices[0]) - 1) * 100
                            
                            # Add to legend items
                            legend_items.append((line, f"{date_str} path: {pct_change:.2f}%"))
                
                # Set the title and labels
                ax_paths.set_title(f"Potential {selected_period}-Day Price Paths After {target_date_str} (Based on Similar Historical Dates)")
                ax_paths.set_xlabel("Date")
                ax_paths.set_ylabel("Price")
                ax_paths.grid(True, alpha=0.3)
                
                # Create custom legend with only path items
                if legend_items:
                    lines, labels = zip(*legend_items)
                    ax_paths.legend(lines, labels, loc='best', fontsize='small')
                
                plt.tight_layout()
                st.pyplot(fig_paths)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
                logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
