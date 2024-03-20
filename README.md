# Market Insight & Forecasting Tool

The Market Insight & Forecasting Tool is a powerful analysis tool designed to provide deep insights into market trends, standardise and compare different time series data, and visualise potential outcomes based on historical data. Utilizing some simple machine learning and statistical techniques, this tool provides both traders and investors a lens through which they can view potential outcomes based on a filtering of historical data using a number of predictive market factors.

## Features

- **Historical Market Data Retrieval:** Fetch historical market data for various tickers, including stock indices, commodities, and cryptocurrencies.
- **Feature Engineering:** Calculate and analyse various financial indicators related to Volume/Momentum/Trend/Volatilty.
- **Similarity Measure:** Filtering for dates with market conditions similar to a user-defined target date, aiding in pattern recognition and prediction.
- **Forward Return Analysis:** Calculate forward returns for selected periods to assess potential future performance.
- **Resampling and KDE Plotting:** Visualise the probability distribution of forward returns, providing a graphical representation of potential outcomes, and potential biases to form trades.

## Accessing the Tool

The Market Insight & Forecasting Tool is hosted on Streamlit, making it easy to access and use through any web browser. Follow the link below to start using the tool:

[Launch Market Insight & Forecasting Tool](https://market-filter-forecaster.streamlit.app/)

## How to Use

### Step 1: Select a Ticker
At the top of the page, you will find a dropdown menu titled "Select A Ticker Symbol". Click on the dropdown to reveal a list of available tickers, which include major stock indices, commodities, and cryptocurrencies. Select the one you wish to analyze.

### Step 2: Set Data Cutoff Date
In the "Analysis Date" input field, select the date up to which you want the data to be considered for the analysis. This tool performs predictive analysis, so the specified date marks the cutoff for historical data used to project the forward returns for the subsequent 5 business days. The analysis will not include data beyond this date to ensure that forecasts are based solely on historical information available up to that point. Please be aware that analysis is viable for business days only; selecting weekends or dates in the future will result in an error message.

### Step 3: Analyse
With your ticker and date selected, click the "Analyse" button to initiate the analysis process. The tool will then process the historical data and generate various financial indicators related to volume, momentum, trend, and volatility.

### Results Visualization
After analysis, the tool will display the "Probability Distribution of 5 day Forward Returns", as shown in the Streamlit layout screenshot. This graph visualizes the potential outcomes and biases for the selected ticker and date. It includes:

- A histogram showing the distribution of simulated forward returns.
- A KDE (Kernel Density Estimate) curve that gives a smooth estimate of the distribution.
- Vertical lines indicating the mean and median of the forward returns.

Use this visual representation to inform your trading decisions, pattern recognition, and potential market predictions.
In an ideal forecast, you want to see a distribution with a significant directional bias, which is indicated by either high positive or negative mean/median returns. Such a bias can signal potential trends across all features and may guide your trading decisions. A more skewed distribution towards positive or negative returns can be interpreted as an indicator of aligned market sentiment, giving you an edge in pattern recognition and market predictions.


<img src="https://github.com/mohidyasin/1st-project/assets/164057664/e0a5caa3-5adc-4014-9983-8be8a0415c3b" width="75%" height="50%">


