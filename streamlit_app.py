import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns

# Configure page
st.set_page_config(page_title="Smart Portfolio Manager", layout="wide")
st.title('AI-Driven Portfolio Management')
st.write("""
### Dynamic Portfolio Optimization using Modern Finance Theory
""")

# List of 20 stocks across different industries with validation
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA',    # Tech
    'JNJ', 'PFE', 'MRK',                       # Healthcare
    'JPM', 'BAC', 'GS',                        # Financials
    'WMT', 'TGT', 'COST',                      # Retail
    'XOM', 'CVX',                              # Energy
    'UNH', 'PG', 'DIS', 'NKE'                  # Diverse sectors
]

def get_valid_tickers(tickers):
    """Filter out invalid/unavailable tickers"""
    valid = []
    for t in tickers:
        try:
            if yf.Ticker(t).info['regularMarketPrice']:
                valid.append(t)
        except:
            continue
    return valid

def fetch_data():
    """Fetch historical stock data with error handling"""
    valid_tickers = get_valid_tickers(STOCK_UNIVERSE)
    if not valid_tickers:
        return pd.DataFrame()
    
    try:
        data = yf.download(
            tickers=valid_tickers,
            period="1y",
            interval="1d",
            group_by='ticker',
            progress=False
        )
        # Safely access Adjusted Close prices
        adj_close = data.xs('Adj Close', level=1, axis=1)
        return adj_close.dropna(axis=1)
    except Exception as e:
        st.error(f"Data retrieval error: {str(e)}")
        return pd.DataFrame()

def calculate_momentum(data):
    """Calculate momentum factor (3-month returns)"""
    returns = data.pct_change().dropna()
    return returns.last('3M').mean()

def calculate_value(data):
    """Calculate value factor (P/E ratio) with error handling"""
    pe_ratios = {}
    for ticker in data.columns:
        try:
            pe = yf.Ticker(ticker).info.get('trailingPE', np.nan)
            pe_ratios[ticker] = pe if not None else np.nan
        except:
            pe_ratios[ticker] = np.nan
    return pd.Series(pe_ratios).replace(np.inf, np.nan).dropna()

def optimize_portfolio(risk_category, data):
    """Optimize portfolio based on risk category"""
    if data.empty:
        return {}
        
    returns = expected_returns.mean_historical_return(data)
    cov_matrix = risk_models.sample_cov(data)
    
    try:
        if risk_category == "Aggressive":
            ef = EfficientFrontier(returns, cov_matrix)
            ef.add_objective(objective_functions.L2_reg)
            ef.max_sharpe()
            weights = ef.clean_weights()
            
        elif risk_category == "Moderate":
            # Factor investing: 50% momentum, 50% value
            momentum = calculate_momentum(data)
            value = calculate_value(data)
            
            # Normalize and combine scores
            momentum_norm = momentum.rank(pct=True)
            value_norm = (1/value.rank(pct=True))  # Lower P/E is better
            
            combined_score = 0.5*momentum_norm + 0.5*value_norm
            selected = combined_score.nlargest(10).index.tolist()
            
            # Optimize on selected stocks
            ef = EfficientFrontier(returns[selected], cov_matrix.loc[selected, selected])
            ef.max_sharpe()
            weights = ef.clean_weights()
            
        else:  # Conservative
            volatility = data.pct_change().std().nsmallest(5)
            weights = {t: 1/len(volatility) for t in volatility.index}
            
        return {k: v for k, v in weights.items() if v > 0.01}
        
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

# Risk Assessment Sidebar
with st.sidebar:
    st.header("Investor Profile")
    investment_horizon = st.selectbox("Investment Horizon", 
        ["Short-term (1-3 years)", "Medium-term (3-5 years)", "Long-term (5+ years)"])
    risk_tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 5)
    experience = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])

# Main application logic
if st.button("Build Optimal Portfolio"):
    with st.spinner("Crunching numbers using Modern Portfolio Theory..."):
        data = fetch_data()
        
        if data.empty:
            st.error("No valid financial data could be retrieved. Please try different assets.")
            st.stop()
            
        # Determine risk category
        if risk_tolerance <= 3:
            risk_category = "Conservative"
            allocation = {"Stocks": 30, "Bonds": 50, "REITs": 15, "Cash": 5}
        elif risk_tolerance <= 7:
            risk_category = "Moderate"
            allocation = {"Stocks": 60, "Bonds": 25, "REITs": 10, "Cash": 5}
        else:
            risk_category = "Aggressive"
            allocation = {"Stocks": 90, "Bonds": 5, "REITs": 0, "Cash": 5}

        # Optimize stock portfolio
        stock_weights = optimize_portfolio(risk_category, data)
        if not stock_weights:
            st.error("Failed to optimize portfolio. Please adjust your parameters.")
            st.stop()
            
        stock_percentage = allocation["Stocks"]
        
        # Create full allocation
        full_allocation = {}
        for stock, weight in stock_weights.items():
            full_allocation[stock] = weight * stock_percentage
        for asset, weight in allocation.items():
            if asset != "Stocks":
                full_allocation[asset] = weight

        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(full_allocation.values(), labels=full_allocation.keys(),
               autopct='%1.1f%%', startangle=90,
               colors=plt.cm.Paired.colors,
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        ax.set_title(f"{risk_category} Portfolio Allocation", fontsize=16)
        
        # Display results
        col1, col2 = st.columns([3, 2])
        with col1:
            st.pyplot(fig)
        with col2:
            st.subheader("Portfolio Construction Strategy")
            st.write(f"""
            **Risk Category:** {risk_category}
            - **Optimization Method:** {'Factor Investing' if risk_category == 'Moderate' else 'Markowitz Optimization'}
            - **Stock Selection:** {len(stock_weights)} companies across {len(set([s.split()[0] for s in stock_weights]))} sectors
            - **Rebalancing Frequency:** {'Quarterly' if risk_category == 'Conservative' else 'Monthly'}
            """)
            
            st.subheader("Key Statistics")
            returns = data[list(stock_weights.keys())].pct_change().mean().dot(list(stock_weights.values()))
            volatility = np.sqrt(np.dot(list(stock_weights.values()), 
                                     np.dot(data[list(stock_weights.keys())].pct_change().cov(), 
                                            list(stock_weights.values()))))
            st.write(f"""
            - **Expected Annual Return:** {returns*252:.1%}
            - **Expected Volatility:** {volatility*np.sqrt(252):.1%}
            - **Sharpe Ratio:** {returns/volatility:.2f}
            """)
            
        st.subheader("Portfolio Composition Details")
        st.dataframe(pd.DataFrame.from_dict(full_allocation, orient='index', 
                      columns=['Allocation (%)']).sort_values(by='Allocation (%)', ascending=False))
        
        st.markdown("""
        **Investment Strategy Details:**
        - **Factor Investing (Moderate):** Combines value (P/E ratios) and momentum (3-month returns) factors
        - **Markowitz Optimization (Aggressive):** Maximizes Sharpe ratio using Modern Portfolio Theory
        - **Conservative Portfolio:** Focuses on low-volatility blue-chip stocks
        """)
