import pandas as pd
import yfinance as yf
import pandas_ta as ta
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from pandas.tseries.offsets import BDay
import mplfinance as fplt

import streamlit as st


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

#########################################################################################################

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



# forward returns
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
    ax.axvline(x=(close_price_on_target_date / close_price_on_target_date - 1) * 100, color='black', linestyle='-', label=f'Close Price on Target       Date ({close_price_on_target_date:.2f})')

    # Set the title to reflect the next business day after the target date and forward return period
    ax.set_title(f' Probability Distribution of {forward_return_period} Forward Returns for {next_day_str}', weight='bold')
    ax.grid(True)
    


def main():
    st.title("Market Insight & Forecasting Tool")

    # Mapping of allowed tickers to their names for better accessibility
    ticker_names = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ Composite',
        '^DJI': 'Dow Jones Industrial Average',
        'CL=F': 'Crude Oil Futures',
        'GC=F': 'Gold Futures',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'BTC-USD': 'Bitcoin'
    }

    # User Inputs
    # User selects ticker from the list of names
    selected_name = st.selectbox("Select A Ticker Symbol:", list(ticker_names.values()), index=0)
    # Find the ticker symbol corresponding to the selected name
    ticker = [symbol for symbol, name in ticker_names.items() if name == selected_name][0]
    
    
    # Calculate the allowed date range: last 3 years, up to today
    max_date = datetime.now() - timedelta(days=1)  # Set max date to yesterday
    min_date = max_date - timedelta(days=365*3)  # Last 3 years

    # Define business days (Monday=0, Sunday=6)
    business_days = [0, 1, 2, 3, 4]  # Weekdays
    
    # Date input with constraints
    target_date = st.date_input("**Analysis Date**", max_value=max_date, min_value=min_date, value=max_date, help = "Data Collected Up To This Date")

    target_date_str = target_date.strftime('%Y-%m-%d')
    valid_date_selected = target_date.weekday() in business_days

    # Display an error message if a weekend is selected, effectively blocking analysis
    if not valid_date_selected:
        st.error("Error: The selected date falls on a weekend. Please choose a weekday.")

    # Button to trigger analysis, disabled if invalid date is selected
    if st.button('Analyse', disabled=not valid_date_selected):
        with st.spinner('Performing Analysis...'):
            perform_analysis(ticker, target_date_str)
            
            
            
def perform_analysis(ticker, target_date_str):
    
    # Fetch data
    with st.spinner('Retrieving Historical Market Data...'):
        
        data = yf.download(ticker, start='2000-01-01', end=datetime.now().strftime('%Y-%m-%d'))

    # Data processing and Feature Engineering
    with st.spinner('Preparing and Organising Data...'):
        
        # Calculate the 14-period RSI
        data['RSI_14'] = ta.rsi(data['Close'], length=14)
        # Calculate the 5-day average of RSI and round it to 2 decimal places
        data['5d_avg_RSI'] = data['RSI_14'].rolling(window=5).mean().round(2)
        # Calculate the percent change in the RSI_14 and round it to 2 decimal places
        data['RSI_14_pct_change'] = data['RSI_14'].pct_change().mul(100).round(2)
        data['RSI_14_Slope'] = calculate_slope(data['RSI_14'], 3)  #Window Length
        data['RSI_14_Slope'] = data['RSI_14_Slope'].round(2) #rounding
        # Calculate the 20-day EMA of the closing price and round it to 2 decimal places
        data['20d_EMA'] = ta.ema(data['Close'], length=20).round(2)
        # Calculate the percentage distance from the closing price to the 10-day EMA and round it to 2 decimal places
        data['Pct_Distance_to_EMA'] = ((data['Close'] - data['20d_EMA']) / data['20d_EMA']).mul(100).round(2)
        data['EMA_Slope'] = calculate_slope(data['20d_EMA'], 3)  #Window Length
        data['EMA_Slope'] = data['EMA_Slope'].round(2) #rounding
        # Calculate True Range (TR) as an intermediate step
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Calculate the 14-day Exponential Moving Average of True Range for ATR
        data['ATR_14'] = true_range.ewm(span=14, adjust=False).mean().round(2)
        # Calculate the rank of the ATR over the past 252 rows
        data['ATR_14_rank'] = data['ATR_14'].rolling(window=252).apply(lambda x: x.rank(pct=True)[-1]).round(2)
        # Volume
        data['Volume_avg_5d_ema'] = ta.ema(data['Volume'], length=5).round(2)
        data['Volume_ema_slope'] = calculate_slope(data['Volume_avg_5d_ema'], 3)  #Window Length
        data['Volume_ema_slope'] = data['Volume_ema_slope'].round(2) #rounding
        data['Volume_avg_5d_ema_rank'] = data['Volume_avg_5d_ema'].rolling(window=252).apply(lambda x: x.rank(pct=True)[-1]).round(2)
        # Convert the index to a datetime 
        data.index = pd.to_datetime(data.index)
        data.dropna(inplace=True)
        
        # Features to standardise
        excluded_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        features = data.columns.difference(excluded_columns)
        window_size = 252
        target_date = pd.to_datetime(target_date_str)
        standardized_data = standardize_data_up_to_date(data, features, window_size, target_date_str)
        
    # Similarity Measure and Forward Returns
    with st.spinner('Analysing Market Trends...'):
        
        # Use the 'features' list to select the relevant columns from the last row corresponding to the target date
        latest_day_data = standardized_data.loc[target_date_str, features].values
        # Call the function with the target date and standardized data
        lorentzian_dates = get_similar_dates_by_dist(standardized_data, target_date, lorentzian_distance, features)
        # Returns
        standardized_data['Returns'] = (standardized_data['Close'].pct_change() * 100).round(2)
        standardized_data['1d_Forward_Return'] = standardized_data['Returns'].shift(-1)  # 1-day forward return
        standardized_data['5d_Forward_Return'] = (((standardized_data['Close'].shift(-4) / standardized_data['Close']) - 1) * 100).round(2)
        standardized_data['20d_Forward_Return'] = (((standardized_data['Close'].shift(-19) / standardized_data['Close']) - 1) * 100).round(2)
        # Assuming lorentzian_dates is defined elsewhere in your code
        lorentzian_dates_only = [date for date, _ in lorentzian_dates]
        # Calculate forward returns for these dates
        forward_returns_list = get_forward_returns(lorentzian_dates_only, standardized_data, target_date)
        # Sort the forward returns list by date
        sorted_forward_returns = sorted(forward_returns_list, key=lambda x: x[0])
        # Convert the sorted forward returns list to a DataFrame
        data_feats_fwd_ret = pd.DataFrame(sorted_forward_returns, columns=['Date', '1D_Forward_Return', '5D_Forward_Return', '20D_Forward_Return'])

        
    # Resampling and KDE Plot
    with st.spinner('Visualising Potential Outcomes...'):
        
        # Set a seed value for reproducibility
        np.random.seed(123)
        # Generate samples for each forward return period
        samples_1D = generate_samples('1D', data_feats_fwd_ret, "1D_Forward_Return")
        samples_5D = generate_samples('5D', data_feats_fwd_ret, "5D_Forward_Return")
        samples_20D = generate_samples('20D', data_feats_fwd_ret, "20D_Forward_Return")
        # Calculate the date that is one business day after the target date
        next_day = target_date + BDay(1)
        # Format the next day date as a string for the plot title
        next_day_str = next_day.strftime('%Y-%m-%d')
        # Extract the "Close" price for the target date
        close_price_on_target_date = standardized_data.loc[target_date, 'Close']
        
        
        # Create a grid for plots
        fig, axs = plt.subplots(1, 1, figsize=(18, 12))
        # Plotting for samples_20D
        plot_samples(samples_20D, '5D', np.mean(samples_5D), np.median(samples_5D), target_date_str, close_price_on_target_date, ticker,                 next_day_str, axs)
        
        # Display plot in Streamlit app
        st.pyplot(fig)          
        




if __name__ == "__main__":
    main()
