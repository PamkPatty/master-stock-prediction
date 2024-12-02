import numpy as np
import pandas as pd
import datetime as dt
import pandas as pd
import streamlit as st
from datetime import date
from rest_api.fetch_data import (get_symbol_data)
from visualizations.plots import (
    beta,
    basic_portfolio,
    display_portfolio_return,
    display_heat_map,
    monte_carlo,
)

st.set_page_config(page_title='Futures')

def load_heading():
    """The function that displays the heading.
        Provides instructions to the user
    """
    with st.container():
        st.title('Stock Forecast App: Futures🚀')
        st.info('This page performs historical portfolio analysis and future analysis with **Monte Carlo Simulation🎰**. Please read the instructions carefully and enjoy!📚')



def get_choices():
    """Prompts the dialog to get the All Choices.

    Returns:
        An object of choices and an object of combined dataframes.
    """
    choices = {}
    user_start_date = date.today()
    yesterday = user_start_date

    # add_selectbox = st.sidebar.selectbox(
    # "How would you like to be contacted?",
    # ("Email", "Home phone", "Mobile phone")
    # )
    
    st.sidebar.header('Query parameters📦')
    with st.expander("🧩Rules"):
        st.sidebar.info("You must enter 1 Index such as SPY, 3 Stock, and 2 Crypto Symbol names. Please refer to Yahoo Finance for a list of applicable ticker symbols.  Type the symbol EXACTLY as provided by Yahoo Finance.")
  
    years_back = st.sidebar.number_input('📆How many years back from today?', min_value=1, max_value=20, value=20)
    if years_back <= 9:
       st.sidebar.warning("The less years back from today you input, the less accuracy will be.🤕")
      
    tickers = st.sidebar.text_input('🎫Enter 1 Index and 3 Stock symbols:', 'SPY,META,TSLA,NVDA')
    tickers_list = tickers.split(",")
    


    crypto_symbols = st.sidebar.text_input('💎Enter 2 Crypto symbols only as below:', 'BTC-USD,ETH-USD')
    # 'SPY,AMZN,TSLA,NVDA,AAPL,BTC-USD,ETH-USD'


    # Set the weights
    weights_str = st.sidebar.text_input('📦Enter the investment weights:', '0.2,0.2 ,0.2,0.2,0.1,0.1')
    weights_list = weights_str.split(",")
    weights = []
    for item in weights_list:
        weights.append(float(item))
    if sum(weights) != 1:
        st.sidebar.error('Check weights, sum of weight must = 1️.😫')


    # Set Initial Investment
    investment = st.sidebar.number_input('💸Enter the initial investment:', min_value=1000, max_value=100000, value=5000)
    # Set the investment forecast_years
    forecast_years = st.sidebar.number_input('⏳Enter the forecast years for the sim:', min_value=1, max_value=10, value=1)
    if forecast_years >= 4:
       st.sidebar.warning("The more years you input, the less accuracy will be.🤕")
    # Set the number of simulations to run_years
    
    sim_runs = st.sidebar.number_input('🐱‍🏍Enter the number of simulations to run', min_value=100, max_value=5000, value=1000)
    st.sidebar.info("We recommend you run 1000 sim runs. 100 is minimum and 5000 is max:")

    # Set the start_date to years_back  
    start_date = user_start_date.replace(year=(yesterday.year - years_back), month=yesterday.month, day=yesterday.day)
    # Set the end_date to yesterday
    end_date = yesterday

    # Every form must have a submit button.
    submitted = st.sidebar.button("Submit")

    symbols = []
    reset = False

    # Reusable Error Button DRY!
    def reset_app(error):
        st.sidebar.error(f"Check The Syntax👀")
        st.sidebar.error(f"{error}!")
        reset = st.sidebar.button("🤬Reset page🤬")

    if submitted:
        # convert  strings to lists
        tickers_list = tickers.split(",")
        weights_list = weights_str.split(",")
        crypto_symbols_list = crypto_symbols.split(",")
        # Create the Symbols List
        symbols.extend(tickers_list)
        symbols.extend(crypto_symbols_list)
        # Convert Weights To Decimals
        weights = []
        for item in weights_list:
            weights.append(float(item))

        # CheckThe User Input
        if len(tickers_list) != 4:
            reset_app('Check Stock tickers🎫')
            st.error('Check Stock tickers🎫')
        if len(crypto_symbols_list) != 2:
            reset_app('Check Crypto tickers💎')
            st.error('Check Crypto tickers💎')
        if sum(weights) != 1:
            reset_app('Check Weights📦')
            st.error('Check Weights📦')

        if reset:
            # Clears all singleton caches:
            tickers = st.sidebar.text_input('🎫Enter 1 Index and 3 Stock symbols:', 'SPY,META,TSLA,NVDA')
            crypto_symbols = st.sidebar.text_input('💎Enter 2 Crypto symbols only as below:', 'BTC-USD,ETH-USD')
            weights_str = st.sidebar.text_input('📦Enter the investment weights:', '0.2,0.2 ,0.2,0.2,0.1,0.1')
            st.experimental_singleton.clear()

        else:    
            # Submit an object with choices
            choices = {
                'user_start_date': user_start_date,
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols,
                'weights': weights,
                'investment': investment,
                'forecast_years': forecast_years,
                'sim_runs': sim_runs
            }
            # Load combined_df
            combined_df = get_symbol_data(choices)
            # return object of objects
            return {
                'choices': choices,
                'combined_df': combined_df
            }


def run():
    """The main function for running the script."""

    load_heading()
    choices = get_choices()
    if choices:     
        beta(choices['combined_df'])
        basic_portfolio(choices['combined_df'])
        display_heat_map(choices['combined_df'])
        display_portfolio_return(choices['combined_df'], choices['choices'])
        with st.spinner('Running Monte Carlo Simulation...'):
            monte_carlo(choices['combined_df'], choices['choices'])
        

if __name__ == "__main__":
    run()