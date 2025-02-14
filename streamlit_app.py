import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns

# Configure page
st.set_page_config(page_title="Smart Portfolio Manager", layout="wide")
st.title('Advanced Portfolio Optimizer')
st.write("""
### Multi-Asset Allocation with Crypto Integration
""")

# Asset universe with sectors
ASSET_UNIVERSE = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOG': 'Tech',
    'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto', 
    'SPY': 'ETF', 'GLD': 'Commodity', 'TLT': 'Bonds',
    'JPM': 'Financial', 'XOM': 'Energy'
}

def fetch_data():
    """Fetch market data with error handling"""
    try:
        data = yf.download(
            list(ASSET_UNIVERSE.keys()),
            period="1y",
            interval="1d",
            group_by='ticker',
            progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            return data.xs('Close', level=1, axis=1).ffill().dropna(axis=1)
        return data['Close'].to_frame().ffill().dropna(axis=1)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_risk_profile(horizon, tolerance, experience):
    """Dynamic risk assessment"""
    scores = {
        'horizon': {'<1y': 1, '1-3y': 2, '3-5y': 3, '5y+': 4}[horizon],
        'tolerance': tolerance,
        'experience': {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}[experience]
    }
    total = scores['horizon']*2 + scores['tolerance'] + scores['experience']
    return "Conservative" if total < 8 else "Moderate" if total < 12 else "Aggressive"

def optimize_portfolio(risk_profile, data):
    """Portfolio optimization engine"""
    if data.empty or len(data.columns) < 2:
        return {}
    
    try:
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        
        if risk_profile == "Aggressive":
            # Crypto allocation constraints
            crypto_assets = [t for t in data.columns if ASSET_UNIVERSE[t] == 'Crypto']
            if crypto_assets:
                ef.add_constraint(lambda w: sum(w[c] for c in crypto_assets) >= 0.25)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.max_sharpe()
            
        elif risk_profile == "Moderate":
            # Factor-based selection
            momentum = data.pct_change(90).mean()
            volatility = data.pct_change().std()
            selected = (momentum / volatility).nlargest(8).index
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.max_sharpe()
            
        else:  # Conservative
            ef.min_volatility()
            ef.add_constraint(lambda w: w <= 0.15)
            
        return ef.clean_weights()
    
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

def plot_growth(weights, data, initial=100000):
    """Monte Carlo growth projection"""
    returns = data.pct_change().dropna().dot(list(weights.values()))
    simulations = 500
    years = 5
    
    # Geometric Brownian Motion parameters
    mu = returns.mean()
    sigma = returns.std()
    daily_returns = np.random.normal(mu/252, sigma/np.sqrt(252), (252*years, simulations))
    
    growth = initial * np.cumprod(1 + daily_returns, axis=0)
    return pd.DataFrame(growth)

# Sidebar - Risk Profiling
with st.sidebar:
    st.header("Investor Profile")
    horizon = st.selectbox("Investment Horizon", ["<1y", "1-3y", "3-5y", "5y+"])
    tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 5)
    experience = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
    risk_profile = calculate_risk_profile(horizon, tolerance, experience)
    st.metric("Your Risk Profile", risk_profile)

# Main Application
if st.button("Generate Portfolio"):
    with st.spinner("Optimizing portfolio..."):
        data = fetch_data()
        if data.empty:
            st.error("Failed to fetch market data")
            st.stop()
            
        weights = optimize_portfolio(risk_profile, data)
        if not weights:
            st.error("Portfolio optimization failed")
            st.stop()
        
        # Calculate metrics
        valid_weights = {k: v for k, v in weights.items() if v > 0.01}
        returns = data[valid_weights.keys()].pct_change().mean().dot(list(valid_weights.values()))
        volatility = np.sqrt(
            np.dot(
                list(valid_weights.values()),
                np.dot(
                    data[valid_weights.keys()].pct_change().cov(),
                    list(valid_weights.values())
                )
            )
        )
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual Holdings
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.pie(valid_weights.values(), labels=valid_weights.keys(),
                    autopct='%1.1f%%', startangle=90)
            ax1.set_title("Asset Allocation")
            st.pyplot(fig1)
            
            # Volatility Display
            st.metric("Annualized Volatility", f"{volatility*np.sqrt(252):.1%}")

        with col2:
            # Sector Allocation
            sector_alloc = pd.Series(valid_weights).groupby(ASSET_UNIVERSE.get).sum()
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.pie(sector_alloc, labels=sector_alloc.index,
                    autopct='%1.1f%%', startangle=90)
            ax2.set_title("Sector Allocation")
            st.pyplot(fig2)
            
            # Crypto Exposure
            crypto_exposure = sum(v for k,v in valid_weights.items() 
                                if ASSET_UNIVERSE.get(k) == 'Crypto')
            st.metric("Crypto Allocation", f"{crypto_exposure:.1%}")

        # Growth Projection
        st.subheader("€100,000 Growth Projection (5 Years)")
        growth = plot_growth(valid_weights, data)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(growth, color='blue', alpha=0.1)
        ax3.plot(growth.median(axis=1), color='red', linewidth=2, label='Median')
        ax3.set_xlabel("Trading Days")
        ax3.set_ylabel("Portfolio Value (€)")
        ax3.legend()
        st.pyplot(fig3)

