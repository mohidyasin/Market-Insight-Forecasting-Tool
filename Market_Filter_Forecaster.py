import pandas as pd
import pandas_ta as ta
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from pandas.tseries.offsets import BDay
import streamlit as st
from tvDatafeed import TvDatafeed, Interval

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
    
    # Convert index to date objects if they're datetime objects
    if isinstance(data.index[0], pd.Timestamp):
        cutoff_date = cutoff_date.date()
        compare_index = [d.date() if hasattr(d, 'date') else d for d in data.index]
    else:
        compare_index = data.index
    
    # Create a mask for dates up to and including the cutoff date
    mask = [d <= cutoff_date for d in compare_index]
    
    # Return the filtered dataframe
    return data.iloc[mask]

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

# Forward returns
def get_forward_returns(dates, df, target_date):
    forward_returns = []
    for date in dates:
        if date in df.index:
            idx = df.index.get_loc(date)
            # Ensure there's enough data to calculate each forward return
            forward_return_1d = df.loc[date, '1d_Forward_Return'] if idx < len(df) - 1 else None
            forward_return_5d = df.loc[date, '5d_Forward_Return'] if idx < len(df) - 4 else None
            forward_return_20d = df.loc[date, '20d_Forward_Return'] if idx < len(df) - 19 else None
            forward_returns.append((date, forward_return_1d, forward_return_5d, forward_return_20d))
    return forward_returns

# Generate samples
def generate_samples(forward_return_period, data_feats_fwd_ret, column_name, num_samples=1000):
    forward_returns = data_feats_fwd_ret[f'{forward_return_period}_Forward_Return'].dropna().astype(float)
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

# Plot samples
def plot_samples(samples, forward_return_period, mean_resampled, median_resampled, target_date_str, close_price_on_target_date, ticker, next_day_str, ax):
    # Compute KDE for the samples
    density = gaussian_kde(samples)
    min_data, max_data = np.min(samples), np.max(samples)
    x = np.linspace(min_data, max_data, 100)  # 100 points for KDE evaluation
    
    # Plot KDE on primary x-axis for returns
    color = 'tab:blue'
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Density', color='black')
    ax.plot(x, density(x), label=f'Close Price ({close_price_on_target_date:.2f})', color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.hist(samples, bins=30, density=True, alpha=0.5, color=color)
    ax.legend(loc='upper left')

    # Plot mean, median
    ax.axvline(x=mean_resampled, color='red', linestyle='--', label=f'Mean Return: {mean_resampled:.2f}%')
    ax.axvline(x=median_resampled, color='green', linestyle='--', label=f'Median Return: {median_resampled:.2f}%')
    ax.legend(loc='upper left')

    # Add a secondary x-axis to show corresponding prices
    ax2 = ax.secondary_xaxis('top', functions=(lambda r: return_to_price(r, close_price_on_target_date),
                                               lambda p: (p / close_price_on_target_date - 1) * 100))
    ax2.set_xlabel(f'Data Based On {target_date_str} Close', color='tab:red')
    ax2.tick_params(axis='x', labelcolor='tab:red')

    # Add a vertical dashed line for the close price on the target date
    ax.axvline(x=(close_price_on_target_date / close_price_on_target_date - 1) * 100, color='black', linestyle='-', label=f'Close Price on Target Date ({close_price_on_target_date:.2f})')

    # Set the title to reflect the next business day after the target date and forward return period
    ax.set_title(f'{ticker} Probability Distribution of {forward_return_period} Forward Returns for {next_day_str}')
    ax.grid(True)

def main():
    st.title("Market 5-Day Forecast Tool")
    
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
    
    # Date selection
    max_date = datetime.now() - timedelta(days=1)  # Yesterday
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
                
                # Filter data to only include dates up to the target date
                data = filter_data_to_date(data, target_date)
                
                st.success(f"Retrieved {len(data)} days of historical data up to {target_date_str}")
                
                # Calculate technical indicators
                data['RSI_14'] = ta.rsi(data['Close'], length=14)
                data['5d_avg_RSI'] = data['RSI_14'].rolling(window=5).mean().round(2)
                data['RSI_14_pct_change'] = data['RSI_14'].pct_change().mul(100).round(2)
                data['RSI_14_Slope'] = calculate_slope(data['RSI_14'], 3)
                data['RSI_14_Slope'] = data['RSI_14_Slope'].round(2)
                data['20d_EMA'] = ta.ema(data['Close'], length=20).round(2)
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
                data['Volume_avg_5d_ema'] = ta.ema(data['Volume'], length=5).round(2)
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
                
                # Calculate returns
                standardized_data['Returns'] = (standardized_data['Close'].pct_change() * 100).round(2)
                standardized_data['1d_Forward_Return'] = standardized_data['Returns'].shift(-1)
                standardized_data['5d_Forward_Return'] = (((standardized_data['Close'].shift(-4) / standardized_data['Close']) - 1) * 100).round(2)
                standardized_data['20d_Forward_Return'] = (((standardized_data['Close'].shift(-19) / standardized_data['Close']) - 1) * 100).round(2)
                
                # Get forward returns for similar dates
                lorentzian_dates_only = [date for date, _ in lorentzian_dates]
                forward_returns_list = get_forward_returns(lorentzian_dates_only, standardized_data, target_date_str)
                sorted_forward_returns = sorted(forward_returns_list, key=lambda x: x[0])
                data_feats_fwd_ret = pd.DataFrame(sorted_forward_returns, columns=['Date', '1D_Forward_Return', '5D_Forward_Return', '20D_Forward_Return'])
                
                # Generate samples
                np.random.seed(123)
                samples_5D = generate_samples('5D', data_feats_fwd_ret, "5D_Forward_Return")
                
                # Calculate next trading day and 5-day forecast end date
                next_day = pd.to_datetime(target_date_str) + BDay(1)
                next_day_str = next_day.strftime('%Y-%m-%d')
                
                # Calculate the end date (5 business days from target date)
                business_days = get_business_dates(target_date_str, 5)
                end_date = business_days[-1].strftime('%Y-%m-%d')
                
                # Get close price on target date
                close_price_on_target_date = standardized_data.loc[target_date_str, 'Close']
                
                # Display enhanced forecast summary
                st.subheader(f"5-Day Forecast ({next_day_str} to {end_date})")
                
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
                col2.metric("Mean Expected Return", f"{np.mean(samples_5D):.2f}%")
                col3.metric("Median Expected Return", f"{np.median(samples_5D):.2f}%")
                
                # Create 5-day forecast plot
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_samples(samples_5D, '5D', np.mean(samples_5D), np.median(samples_5D), 
                             target_date_str, close_price_on_target_date, symbol, next_day_str, ax)
                st.pyplot(fig)
                
                # Show historical price paths
                st.subheader("Historical Price Paths After Similar Dates")
                
                # Calculate future dates for projection (5 business days)
                future_dates = pd.date_range(start=pd.to_datetime(target_date_str), periods=6, freq='B')
                
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
                        
                        # Make sure we have at least 5 more days of data after this date
                        if idx + 5 < len(original_data):
                            # Extract the 5-day future path
                            future_dates_hist = original_data.index[idx:idx+6]
                            future_prices = original_data['Close'].iloc[idx:idx+6].values
                            
                            # Normalize the path to start at the same point as the target date close price
                            price_ratio = chart_data.iloc[target_date_idx]['Close'] / future_prices[0]
                            normalized_prices = future_prices * price_ratio
                            
                            # Plot the future path
                            path_dates = pd.date_range(start=chart_data.index[target_date_idx], periods=6, freq='B')
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
                ax_paths.set_title(f"Potential 5-Day Price Paths After {target_date_str} (Based on Similar Historical Dates)")
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

if __name__ == "__main__":
    main()
