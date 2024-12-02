import streamlit as st
import yfinance as yf
import yahoo_fin.stock_info as si
import plotly.graph_objects as go
import base64
import datetime

# Streamlit page configuration
st.set_page_config(page_title='Company Info')

# Sidebar
st.sidebar.subheader('Query parameters📦')
ticker_list = si.tickers_sp500()
tickerSymbol = st.sidebar.text_input("🎫Select ticker:", "MSFT")

start_date = st.sidebar.date_input("📆Start date:", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("🏁End date:", datetime.date(2024, 11, 1))

# Retrieve ticker data
tickerData = yf.Ticker(tickerSymbol)  # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)  # Historical prices

# Application Title
try:
    string_name = tickerData.info['longName']
    st.title(string_name + "🌌")
except KeyError:
    st.title("Company Information🌌")

# Display Company Info
def show_company_info(tickerSymbol):
    company = yf.Ticker(tickerSymbol)
    info = company.info
    try:
        st.write(f"**🌐Website:** {info['website']}")
        st.write(f"**💹Current Price:** {info['currentPrice']}💲")
        st.title("**About Company**")
        st.write(f"**🚀Sector:** {info['sector']}")
        st.write(f"**🏭Industry:** {info['industry']}")
        st.write(f"**🌎Country:** {info['country']}")
        st.write(f"**🏢City:** {info['city']}")
        st.write(f"**📞Phone:** {info['phone']}")
    except KeyError:
        st.warning("Some company information could not be retrieved.")



# Main function
def main():
    show_company_info(tickerSymbol)

if __name__ == '__main__':
    main()

# Business Summary
try:
    st.title("**Business Summary🧠**")
    company = yf.Ticker(tickerSymbol)
    info = company.info
    st.write(f"**🤼Full Time Employees:** {info['fullTimeEmployees']}")
    string_summary = info.get('longBusinessSummary', "No summary available.")
    st.info(string_summary)
except KeyError:
    st.warning("Business summary information is not available.")

# Candlestick Chart with Moving Averages
st.header('**Candlestick Chart with Moving Averages📊**')

if not tickerDf.empty and len(tickerDf) > 20:
    # Calculate moving averages
    tickerDf['MA20'] = tickerDf['Close'].rolling(window=20).mean()
    tickerDf['MA50'] = tickerDf['Close'].rolling(window=50).mean()

    # Create the candlestick chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=tickerDf.index,
        open=tickerDf['Open'],
        high=tickerDf['High'],
        low=tickerDf['Low'],
        close=tickerDf['Close'],
        name='Candlesticks',
    ))

    # Add moving average lines
    fig.add_trace(go.Scatter(
        x=tickerDf.index, y=tickerDf['MA20'], 
        mode='lines', 
        line=dict(color='blue', width=1.5), 
        name='20-Day MA'
    ))
    fig.add_trace(go.Scatter(
        x=tickerDf.index, y=tickerDf['MA50'], 
        mode='lines', 
        line=dict(color='orange', width=1.5), 
        name='50-Day MA'
    ))

    # Customize layout
    fig.update_layout(
        title=f"Chart for {tickerSymbol}",
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig)
else:
    st.warning("Not enough data to generate the chart. Please adjust the date range.")

# Ticker Data Display
st.header('**Ticker data🎫**')

# Format the data
tickerDf.reset_index(inplace=True)  # Ensure 'Date' is a column
tickerDf['Date'] = tickerDf['Date'].dt.strftime('%Y-%m-%d')  # Format Date
tickerDf = tickerDf.drop(columns=['Dividends', 'Stock Splits'])  # Drop unwanted columns

st.write(tickerDf)

# CSV File Download
def filedownload(tickerDf):
    csv = tickerDf.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="ticker.csv">📁Download CSV File</a>'
    return href

st.markdown(filedownload(tickerDf), unsafe_allow_html=True)
