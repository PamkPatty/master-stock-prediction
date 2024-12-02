import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='Compare')
st.title('Stock Forecast App: Compare🏆')

tickers = si.tickers_sp500()

dropdown = st.multiselect(label='🎫Pick your tickers symbols:', 
                          options=tickers,
                          default=("AAPL","NVDA","META","AMD","AMZN","TSLA"))

st.info("Please, refer to Yahoo Finance for a ticker list of **S&P 500🎫** applicable ticker symbols.  Type the symbol **EXACTLY** as provided by Yahoo Finance.")
start = st.date_input('📆Start:', value=pd.to_datetime('2000-1-1'))
end = st.date_input('🏁End:', value=pd.to_datetime('today'))

metrics = ['P/E Ratio', 'Dividend Yield']  # Add more metrics as needed
selected_metric = st.selectbox('🔬Select a metric to compare:', metrics)

def relativeret(df):
    rel = df.pct_change()
    cumret = (1 + rel).cumprod() - 1
    cumret = cumret.fillna(0)
    return cumret

def calculate_performance_metrics(df):
    metrics = {
        'Average Return': df.mean(),
        'Standard Deviation': df.std(),
        'Maximum Drawdown': df.min(),
        'Sharpe Ratio': df.mean() / df.std()
    }
    return pd.DataFrame(metrics)

def calculate_stock_statistics(df):
    statistics = {
        'Mean': df.mean(),
        'Median': df.median(),
        'Minimum': df.min(),
        'Maximum': df.max(),
        '25th Percentile': np.percentile(df, 25),
        '75th Percentile': np.percentile(df, 75)
    }
    return pd.DataFrame(statistics)

def compare_metrics(dropdown, selected_metric):
    data = {}
    for ticker in dropdown:
        if selected_metric == 'P/E Ratio':
            value = si.get_quote_table(ticker)['PE Ratio (TTM)']
        elif selected_metric == 'Dividend Yield':
            value = si.get_quote_table(ticker)['Forward Dividend & Yield']
        # Add more metric comparisons as needed
        data[ticker] = value
    return pd.DataFrame(data, index=[selected_metric])

def plot_stock_prices(dropdown, start, end):
    fig, ax = plt.subplots(figsize=(10, 8))
    for ticker in dropdown:
        data = yf.download(ticker, start, end)['Adj Close']
        ax.plot(data, label=ticker)
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.header('Stock Prices📈')
    with st.expander("🧩Stock Prices?"):
        st.markdown('**The Stock Prices** section in the provided code displays the historical prices of the selected stocks. It allows users to analyze the trends and patterns in the stock prices over time.')
    st.pyplot(fig)

if len(dropdown) > 0:
    df = yf.download(dropdown, start, end)['Adj Close']
    returns_df = relativeret(df)
    st.header('Returns of 👉{}👈'.format(dropdown))
    with st.expander("🧩Returns of Tickers?"):
        st.markdown('**The Returns of Tickers** section in the provided code displays the line chart of the returns of the selected stocks over time. It allows users to analyze the historical performance of the stocks based on their returns.')
    st.line_chart(returns_df)

    # Plot stock prices
    plot_stock_prices(dropdown, start, end)

    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()

    # Create heatmap
    st.header('Correlation Heatmap🧊')
    with st.expander("🧩Correlation Heatmap?"):
        st.markdown('**The Correlation Heatmap** in the provided code is used to analyze the relationships between the returns of the selected stocks. It helps to understand how the returns of different stocks move in relation to each other.')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)


    # Calculate performance metrics
    performance_summary = calculate_performance_metrics(returns_df)
    st.header('Stock Performance Summary🐱‍👤')
    with st.expander("🧩Stock Performance Summary?"):
        st.markdown('**The Stock Performance Summary** section in the provided code calculates and displays various performance metrics for the selected stocks. It provides a summary of the historical performance of the stocks based on these metrics.')
    st.table(performance_summary)

    # Calculate stock statistics
    statistics_summary = calculate_stock_statistics(returns_df)
    st.header('Stock Statistics🐱‍💻')
    with st.expander("🧩Stock Statistics"):
        st.markdown('**The Stock Statistics** section in the provided code calculates and displays various statistical measures for the returns of the selected stocks. These statistics provide insights into the distribution and characteristics of the returns.')
    
    st.table(statistics_summary)

    # Compare selected metric
    comparison_table = compare_metrics(dropdown, selected_metric)
    st.header('Comparison: {}'.format(selected_metric) + "🐱‍🏍")
    with st.expander("🧩Comparison Clue"):
        st.markdown('**1️⃣ The P/E Ratio (TTM)** stands for Price-to-Earnings Ratio (TTM), which is a financial metric used to evaluate the valuation of a company`s stock. It is calculated by dividing the current market price per share by the earnings per share (EPS) over the trailing twelve months (TTM).')
        st.markdown('**2️⃣ The Forward Dividend & Yield** metric provides information about the dividend payments and dividend yield for a stock. Dividends are the portion of a company`s earnings that are distributed to shareholders as a return on their investment.')
    st.table(comparison_table)