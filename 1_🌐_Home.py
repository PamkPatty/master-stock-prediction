import streamlit as st
from datetime import date
import requests
from bs4 import BeautifulSoup
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import seaborn as sns
import pandas as pd
from plotly.subplots import make_subplots

# Функція для отримання історичних даних
def STOCK(ticker): 
    return yf.Ticker(ticker).history(period="max")

st.set_page_config(page_title='Home')
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}
st.title('Stock Forecast App📡')

selected_stock = st.text_input("👇Input ticker dataset for prediction:", "MSFT")
st.info("Please, refer to Yahoo Finance for a ticker list of **S&P 500🎫** applicable ticker symbols. Type the symbol **EXACTLY** as provided by Yahoo Finance.")

# Current Stock value:
NEW_LINK = 'https://finance.yahoo.com/quote/{}'.format(selected_stock)

tickerData = yf.Ticker(selected_stock)  # Get ticker data
# App title
string_name = tickerData.info['longName']
st.title(string_name + "🌌")

full_page_stock = requests.get(NEW_LINK, headers=headers)
soup = BeautifulSoup(full_page_stock.content, 'html.parser')

stock_price = soup.findAll("fin-streamer", {"class": "livePrice yf-1tejb6", "data-testid": "qsp-price"})
stock_price_change = soup.findAll("fin-streamer", {"class": "priceChange yf-1tejb6", "data-testid": "qsp-price-change"})
stock_change_percent = soup.findAll("fin-streamer", {"class": "priceChange yf-1tejb6", "data-testid": "qsp-price-change-percent"})

st.subheader(stock_price[0].text.replace(",", "") + "💲")
st.text("🙊Price changed: " + stock_price_change[0].text + "💲")
st.text("🙉Percentage: " + stock_change_percent[0].text)

# Date selection
START = st.date_input("📆Start date:", date(2000, 1, 1))
TODAY = date.today().strftime("%Y-%m-%d")

# Year range:
n_years = st.slider('⏳Day range of prediction:', 1, 365, 30)
period = n_years

if n_years >= 91:
    st.warning("The more days you select, the less accuracy will be.🤕")

if st.button("Start Analysis"):
    data = STOCK(selected_stock)
    data.reset_index(inplace=True)

    # Форматування стовпця Date
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

    # Видалення непотрібних стовпців
    data = data.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')

    st.subheader('Raw [' + selected_stock + '] Dataset🥩')
    data_load_state = st.success('Dataset Uploaded successfully!✅')
    with st.expander("👀Check it Out"):
        st.write(data.tail())
    with st.expander("🤔Any missing Values?"):
        st.write(data.isnull().sum())
        st.success('We don’t have any missing values!✅')

    # Аналіз обсягу торгів відносно ціни
    st.subheader('Volume VS Price 📉')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Close'], name='Close'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Volume'], name='Volume', marker_color='red'),
        secondary_y=True,
    )
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Close Stock Price', secondary_y=False)
    fig.update_yaxes(title_text='Volume', secondary_y=True)
    st.plotly_chart(fig)

    # Predict forecast with Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    new_forecast = forecast

    # Show and plot forecast
    with st.spinner('Loading data Forecast data 🎱...'):
        st.subheader('Forecast data 🎱')
        with st.expander("👀Check it Out"):
            st.write(forecast.tail())

    # Show last forecasted value
    last_forecasted_date = forecast.iloc[-1]['ds'].date()
    last_forecasted_value = forecast.iloc[-1]['yhat']

    # FBProphet Plot
    st.subheader('FBProphet Plot 🎯')
    with st.expander("💰What price will be?"):
        st.success(f'Stock price will be {last_forecasted_value:.2f}💲 after {period} day(s) on {last_forecasted_date}📅')
    fig1 = plot_plotly(m, forecast)

    st.subheader("Forecast components😴")
    with st.expander("🧩Clue what Graphs shows"):
        st.markdown('1️⃣ Graph shows information about the trend.')
        st.markdown('2️⃣ Graph shows information about the weekly trend.')
        st.markdown('3️⃣ Graph gives us information about the annual tenure.')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # Update layout for the fig1
    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price"
    )

    # Обмежуємо на останні 10 років
    last_date_forecast = forecast['ds'].iloc[-1]
    five_years_ago = last_date_forecast - pd.DateOffset(years=10)
    
    fig1.update_layout(xaxis_range=[five_years_ago, last_date_forecast])  
    st.plotly_chart(fig1)

    # Split data into train and test sets
    train_size = int(len(df_train) * 0.7)
    train_data = df_train[:train_size]
    test_data = df_train[train_size:]

    # Make predictions on test data
    results = []

    col1, col2 = st.columns(2)
    if not test_data.empty:
        forecast = m.predict(test_data)

        # Calculate Prophet's Test MAE, RMSE, and R-squared
        mae = mean_absolute_error(test_data['y'], forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(test_data['y'], forecast['yhat']))
        mse = mean_squared_error(test_data['y'], forecast['yhat'])
        r2 = r2_score(test_data['y'], forecast['yhat'])

        with col1:
            # Show Prophet's Test Metrics
            st.subheader("Prophet's Test Metrics🧪")

            st.write("👁‍🗨Prediction trust factor:")
            if r2 >= 0.85:
                st.success('📗 - High level of trust factor.')
            elif r2 >= 0.75:
                st.info('📘 - Good level of trust factor.')
            elif r2 >= 0.7:
                st.warning('📒 - Satisfactory level of trust factor.')
            elif r2 >= 0.5:
                st.error('📙 - Low level of trust factor.')
            elif r2 >= -1:
                st.error('📕 - Very low level of trust factor.')

            st.write("R-squared:", r2)
            st.write("MAE (Mean Absolute Error):", mae)
            st.write("MSE (Mean Squared Error):", mse)
            st.write("RMSE (Root Mean Squared Error):", rmse)

        with col2:

            # Таблиця передбачених даних
            st.subheader("Prophet's Foresast🎱")
            forecast_table = new_forecast[['ds', 'yhat']]
            forecast_table.rename(columns={'ds': 'Date', 'yhat': 'Predicted Price'}, inplace=True)
            forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(forecast_table)


    else:
        st.warning("No test data available for the specified date range.")
