import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sys

st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")
st.write(f"Pandas version: {pd.__version__}")
st.write(f"NumPy version: {np.__version__}")
st.write(f"yfinance version: {yf.__version__}")

def main():
    st.title("Minimal Streamlit App Test")
    
    ticker = "^GSPC"
    data = yf.download(ticker, start="2023-01-01", end="2023-12-31")
    
    st.write(f"Data for {ticker}:")
    st.dataframe(data.head())

if __name__ == "__main__":
    main()
