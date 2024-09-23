import streamlit as st
import pandas as pd
import yfinance as yf

def main():
    st.title("Minimal Streamlit App Test")
    
    ticker = "^GSPC"
    data = yf.download(ticker, start="2023-01-01", end="2023-12-31")
    
    st.write(f"Data for {ticker}:")
    st.dataframe(data.head())

if __name__ == "__main__":
    main()
