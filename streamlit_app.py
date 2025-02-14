import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns

# Configure page
st.set_page_config(page_title="Advanced Portfolio Manager", layout="wide")
st.title('Intelligent Wealth Builder')
st.write("""
### Crypto-Enhanced Portfolio Optimization with Growth Projections
""")

# Enhanced universe with crypto assets
STOCK_UNIVERSE = {
    'AAPL': 'Technology', 
    'MSFT': 'Technology',
    'BTC-USD': 'Crypto',
    'ETH-USD': 'Crypto',
    'BNB-USD': 'Crypto',
    'JPM': 'Financial',
    'GS': 'Financial',
    'SPY': 'ETF',
    'GLD': 'Commodity',
    'TLT': 'Bonds'
}

def fetch_data():
    """Fetch data with crypto validation"""
    try:
        data = yf.download(
            list(STOCK_UNIVERSE.keys()),
            period="1y",
            interval="1d",
            group_by='ticker',
            progress=False
        )
        
        adj_close = data.xs('Close', level=1, axis=1) if isinstance(data.columns, pd.MultiIndex) else data['Close']
        return adj_close.ffill().dropna(axis=1)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def optimize_portfolio(risk_profile, data):
    """Enhanced optimization with crypto enforcement"""
    if data.empty: return {}
    
    try:
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        ef = EfficientFrontier(mu, S)
        
        if risk_profile == "Aggressive":
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            crypto_tickers = [t for t in data.columns if STOCK_UNIVERSE[t] == 'Crypto']
            if crypto_tickers:
                ef.add_constraint(lambda w: sum(w[c] for c in crypto_tickers) >= 0.25)
            ef.max_sharpe()
        
        weights = ef.clean_weights()
        return {k: v for k, v in weights.items() if v > 0.01}
    
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

def calculate_growth(weights, data, initial=100000):
    """Monte Carlo growth projection"""
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(list(weights.values()))
    
    # Simulation parameters
    days = 252 * 5  # 5 years
    simulations = 100
    
    # Generate random returns
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    daily_returns = np.random.normal(mu/252, sigma/np.sqrt(252), (days, simulations))
    
    # Calculate growth paths
    growth = initial * np.exp(np.cumsum(daily_returns, axis=0))
    return growth

# Risk Profile Interface
risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])

if st.button("Generate Portfolio"):
    data = fetch_data()
    if data.empty: st.stop()
    
    weights = optimize_portfolio(risk_profile, data)
    if not weights: st.stop()
    
    # Calculate metrics
    returns = data[weights.keys()].pct_change().mean().dot(weights.values())
    volatility = np.sqrt(np.dot(list(weights.values()), 
                        np.dot(data[weights.keys()].pct_change().cov(), 
                               list(weights.values()))))
    
    # Visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Individual Holdings Pie Chart
        fig1, ax1 = plt.subplots()
        ax1.pie(weights.values(), labels=weights.keys(),
                autopct='%1.1f%%', startangle=90)
        ax1.set_title("Individual Holdings Allocation")
        st.pyplot(fig1)
        
    with col2:
        # Sector Allocation
        sector_allocation = pd.Series(weights).groupby(STOCK_UNIVERSE.get).sum()
        fig2, ax2 = plt.subplots()
        ax2.pie(sector_allocation, labels=sector_allocation.index,
                autopct='%1.1f%%', startangle=90)
        ax2.set_title("Sector Allocation")
        st.pyplot(fig2)
        
    with col3:
        st.metric("Expected Annual Return", f"{returns*252:.1%}")
        st.metric("Portfolio Volatility", f"{volatility*np.sqrt(252):.1%}")
        crypto_exposure = sum(weights[t] for t in weights if STOCK_UNIVERSE[t] == 'Crypto')
        st.metric("Crypto Exposure", f"{crypto_exposure:.1%}")

    # Growth Projection
    st.subheader("€100,000 Growth Projection")
    growth_paths = calculate_growth(weights, data)
    fig3, ax3 = plt.subplots()
    ax3.plot(growth_paths, alpha=0.1, color='b')
    ax3.plot(pd.DataFrame(growth_paths).mean(axis=1), color='r', lw=2)
    ax3.set_xlabel("Years")
    ax3.set_ylabel("Portfolio Value (€)")
    ax3.set_title("5-Year Growth Projection")
    st.pyplot(fig3)
