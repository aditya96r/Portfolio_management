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
### Dynamic Portfolio Optimization using Modern Portfolio Theory
""")

# Enhanced stock universe with sector information
STOCK_UNIVERSE = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOG': 'Communication',
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
    'JPM': 'Financial', 'BAC': 'Financial', 'GS': 'Financial',
    'WMT': 'Consumer Staples', 'TGT': 'Consumer Staples', 'COST': 'Consumer Staples',
    'XOM': 'Energy', 'CVX': 'Energy', 'UNH': 'Healthcare',
    'PG': 'Consumer Staples', 'DIS': 'Communication', 'NKE': 'Consumer Discretionary'
}

def get_sector(ticker):
    """Get sector information with fallback"""
    return STOCK_UNIVERSE.get(ticker, 'Other')

def fetch_data():
    """Fetch historical data with error handling"""
    valid_tickers = [t for t in STOCK_UNIVERSE.keys() if yf.Ticker(t).history(period='1mo').shape[0] > 0]
    
    if not valid_tickers:
        st.error("No valid tickers available")
        return pd.DataFrame()
    
    try:
        data = yf.download(list(STOCK_UNIVERSE.keys()), period="1y", interval="1d")['Adj Close']
        return data.dropna(axis=1)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def optimize_portfolio(risk_profile, data):
    """Portfolio optimization using different theories"""
    if data.empty or len(data.columns) < 2:
        return {}
    
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    
    try:
        if risk_profile == "Conservative":
            # Minimum Variance Portfolio (Markowitz)
            ef = EfficientFrontier(mu, S)
            ef.min_volatility()
            
        elif risk_profile == "Moderate":
            # Factor-based optimization
            momentum = data.pct_change().mean()
            value = data.iloc[-1] / data.iloc[-252]  # 1-year price ratio
            
            # Combine factors
            combined = 0.5*momentum.rank() + 0.5*value.rank()
            selected = combined.nlargest(10).index
            
            # Optimize on selected stocks
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.max_sharpe()
            
        else:  # Aggressive
            # Classic Markowitz optimization
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg)
            ef.max_sharpe()
            
        weights = ef.clean_weights()
        return {k: v for k, v in weights.items() if v > 0.01}
    
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        return {}

def create_industry_chart(weights):
    """Create pie chart by industry sectors"""
    sector_allocation = {}
    for ticker, weight in weights.items():
        sector = get_sector(ticker)
        sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
    
    sectors = list(sector_allocation.keys())
    allocations = list(sector_allocation.values())
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(allocations, labels=sectors, autopct='%1.1f%%',
           colors=plt.cm.tab20.colors,
           wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
    ax.set_title("Industry Allocation")
    return fig

# Risk Profile Selection
risk_profile = st.sidebar.selectbox(
    "Select Risk Profile",
    ["Conservative", "Moderate", "Aggressive"],
    index=1
)

if st.button("Generate Portfolio"):
    with st.spinner("Optimizing portfolio..."):
        data = fetch_data()
        if data.empty:
            st.stop()
        
        weights = optimize_portfolio(risk_profile, data)
        if not weights:
            st.error("Optimization failed")
            st.stop()
        
        # Calculate statistics using proper alignment
        returns = data[list(weights.keys())].pct_change().mean()
        portfolio_return = np.dot(list(weights.values()), returns)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Composition")
            fig = create_industry_chart(weights)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Portfolio Details")
            st.write(f"**Risk Profile:** {risk_profile}")
            st.write(f"**Number of Assets:** {len(weights)}")
            st.write(f"**Expected Annual Return:** {portfolio_return*252:.1%}")
            
            st.subheader("Top Holdings")
            for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"{ticker} ({get_sector(ticker)}): {weight:.1%}")

        st.subheader("Portfolio Theory Used")
        if risk_profile == "Conservative":
            st.markdown("""
            **Minimum Variance Portfolio (Markowitz)**
            - Minimizes portfolio volatility
            - Suitable for risk-averse investors
            - Focuses on low-volatility assets
            """)
        elif risk_profile == "Moderate":
            st.markdown("""
            **Factor-Based Optimization**
            - Combines momentum and value factors
            - Balances growth and stability
            - Targets risk-adjusted returns
            """)
        else:
            st.markdown("""
            **Maximum Sharpe Ratio (Markowitz)**
            - Maximizes return per unit of risk
            - Uses L2 regularization for stability
            - Aggressive growth orientation
            """)
