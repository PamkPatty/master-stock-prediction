
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from MCForecastTools import MCSimulation
import random


def beta(stock_df):
    """Uses the stock dataframe to calculate and display the betas of the assets.
    """


    # Creates an empty list for the betas
    beta_list = []
    # Creates a covariance variable
    covariance = 0.0
    # Creates an empty list for the covariance
    covariance_list = []
    # Calculates the daily returns of the inputted dataframe
    daily_returns = stock_df['stock_df'].dropna().pct_change()
    columns = daily_returns.columns.tolist()
    
    for each_column in columns:
        # Don't calculate the index's Beta
        if each_column != columns[0]:
            # Calculates the covariance for each combination of assets
            covariance = daily_returns[each_column].cov(daily_returns[columns[0]])
            # Calculates the variance of each asset
            variance = daily_returns[columns[0]].var()
            # Uses the variance and covariance to calculate the beta
            calc_beta = covariance / variance
            # Appends the beta values to a list
            beta_list.append(calc_beta)
            # Appends the covariance valyes to a list
            covariance_list.append(covariance)
        else:
            # Assign value of 1.0 to index Beta
            beta_list.append(1.0)
            covariance_list.append(1.0)

        # Formats the results
        beta = {'Assets':columns, 'Beta':beta_list}
    # Creates a header for streamlit
    st.subheader('Beta of Assets compared to Index📊')
    with st.expander("🧩Beta?"):
        st.info('**The Beta of an Asset compared to an Index** is an important measure in finance and investing. It provides insights into the relationship between an asset`s price movements and the broader market, helping investors make informed decisions about their investments.')
    st.dataframe(beta)


def basic_portfolio(stock_df):
    """Uses the stock dataframe to graph the normalized historical cumulative returns of each asset.
    """
    # Calculates the daily returns of the inputted dataframe
    daily_return = stock_df['stock_df'].dropna().pct_change()
    # Calculates the cumulative return of the previously calculated daily return
    cumulative_return = (1+ daily_return).cumprod()

    # Creates the title for streamlit
    st.subheader('Portfolio historical normalized cumulative returns📈')
    # Graphs the cumulative returns
    st.line_chart(cumulative_return)


def display_heat_map(stock_df):
    """Uses the stock dataframe to calculate the correlations between the different assets and display them as a heatmap.
    """
    # Calcuilates the correlation of the assets in the portfolio
    price_correlation = stock_df['stock_df'].corr()

    # Creates the title for streamlit
    st.subheader('Heatmap Correlation🧊')
    with st.expander("🧩Correlation Heatmap?"):
        st.markdown('**The Correlation Heatmap** in the provided code is used to analyze the relationships between the returns of the selected stocks. It helps to understand how the returns of different stocks move in relation to each other.')

    # Generates a figure for the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(price_correlation, ax=ax)
    # Displays the heatmap on streamlit
    st.write(fig)
    # Creates a header for the correlation data
    with st.expander("🧩Correlation Data"):
        st.dataframe(price_correlation)
    # Displays the correlation data on streamlit
    


def display_portfolio_return(stock_df, choices):
    """Uses the stock dataframe and the chosen weights from choices to calculate and graph the historical cumulative portfolio return.
    """
    user_start_date, start_date, end_date, symbols, weights, investment, forecast_years, sim_runs  = choices.values()
    
    # Calculates the daily percentage returns of the 
    daily_returns = stock_df['stock_df'].pct_change().dropna()
    # Applies the weights of each asset to the portfolio
    portfolio_returns = daily_returns.dot(weights)
    # Calculates the cumulative weighted portfolio return
    cumulative_returns = (1 + portfolio_returns).cumprod()
    # Calculates the cumulative profit using the cumulative portfolio return
    cumulative_profit = investment * cumulative_returns

    # Graphs the result, and displays it with a header on streamlit
    st.subheader('Portfolio Historical cumulative returns based on inputs💼')
    with st.expander("🧩Show table data"):
        st.write(cumulative_profit.describe())
    st.line_chart(cumulative_profit)



def monte_carlo(mc_data_df, choices):
    """Uses the stock dataframe and the chosen weights from choices to run a monte carlo simulation forecasting the portfolio returns, 
        generate and display a line chart of the results, summary statistics, and upper and lower bounds for future value.
    """

    user_start_date, start_date, end_date, symbols, weights, investment, forecast_years, sim_runs  = choices.values()
    
    # Setting the parameters for the monte carlo simulation
    simulation = MCSimulation(
    portfolio_data = mc_data_df['mc_data_df'],
    weights = weights,
    num_simulation = sim_runs,
    num_trading_days = 365 * forecast_years,
    )

    # Running the monte carlo simulation to calculate cumulative returns of the given time period.
    summary_results = simulation.calc_cumulative_return()
    st.subheader(f'Portfolio Simulation Summary for {forecast_years} year(s)🧶')
    # Graphing the results of the monte carlo simulation as a line chart
    st.line_chart(summary_results)

    # Summarizing the results of the simulation
    simulation_summary = simulation.summarize_cumulative_return()
    
    # Calculating the upper and lower returns based on the summary statistics
    ci_lower_cumulative_return = round(simulation_summary[8] * investment, 2)
    ci_upper_cumulative_return = round(simulation_summary[9] * investment, 2)

    
    # Display the result of the calculations with descriptive text
    st.write(""f"Initial investment of 💲{investment} over the next {forecast_years} years might result within the range of 💲{ci_lower_cumulative_return} and 💲{ci_upper_cumulative_return}")
    with st.expander("🧩Simulation Data"):
        st.dataframe(simulation_summary)