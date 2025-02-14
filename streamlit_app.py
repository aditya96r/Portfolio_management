import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns

# Configure page
st.set_page_config(page_title="Smart Portfolio Manager", layout="wide")
st.title('AI-Driven Wealth Builder')
st.write("""
### Next-Gen Portfolio Optimization with Crypto Allocation
""")

# Enhanced asset universe with crypto
ASSET_UNIVERSE = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOG': 'Tech',
    'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto', 'BNB-USD': 'Crypto',
    'SPY': 'ETF', 'GLD': 'Commodity', 'TLT': 'Bonds',
    'JPM': 'Financial', 'XOM': 'Energy'
}

def fetch_data():
    """Robust data fetcher with crypto support"""
    try:
        data = yf.download(
            list(ASSET_UNIVERSE.keys()),
            period="2y",
            interval="1d",
            group_by='ticker',
            progress=False
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data.xs('Close', level=1, axis=1)
        else:
            adj_close = data['Close'].to_frame()
            
        return adj_close.ffill().dropna(axis=1)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_risk_profile(horizon, tolerance, experience):
    """Dynamic risk scoring system"""
    risk_score = {
        'horizon': {'<1y': 1, '1-3y': 2, '3-5y': 3, '5y+': 4}[horizon],
        'tolerance': tolerance,
        'experience': {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}[experience]
    }
    total = risk_score['horizon'] * 2 + risk_score['tolerance'] + risk_score['experience']
    return "Conservative" if total < 8 else "Moderate" if total < 12 else "Aggressive"

def optimize_portfolio(risk_profile, data):
    """Crypto-enhanced optimization engine"""
    if data.empty: return {}
    
    try:
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        
        if risk_profile == "Aggressive":
            # Force 25-40% crypto allocation
            crypto_assets = [t for t in data.columns if ASSET_UNIVERSE[t] == 'Crypto']
            if crypto_assets:
                ef.add_constraint(lambda w: sum(w[c] for c in crypto_assets) >= 0.25)
                ef.add_constraint(lambda w: sum(w[c] for c in crypto_assets) <= 0.4)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.max_sharpe()
            
        elif risk_profile == "Moderate":
            # Balanced factor approach
            momentum = data.pct_change(90).mean()
            volatility = data.pct_change().std()
            factor_score = momentum / volatility
            selected = factor_score.nlargest(8).index
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.max_sharpe()
            
        else:  # Conservative
            ef.min_volatility()
            ef.add_constraint(lambda w: w <= 0.15)  # Diversification
            
        weights = ef.clean_weights()
        return {k: v for k, v in weights.items() if v > 0.01}
    
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

def growth_projection(weights, data, initial=100000):
    """Monte Carlo simulation for growth"""
    returns = data.pct_change().dropna().dot(list(weights.values()))
    simulations = 100
    years = 5
    days = years * 252
    
    # Geometric Brownian Motion
    mu = returns.mean()
    sigma = returns.std()
    daily_returns = np.random.normal(mu/252, sigma/np.sqrt(252), (days, simulations))
    
    growth = initial * np.cumprod(1 + daily_returns, axis=0)
    return growth

# Risk Profiling Interface
with st.sidebar:
    st.header("Investor Profile")
    horizon = st.selectbox("Investment Horizon", ["<1y", "1-3y", "3-5y", "5y+"])
    tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 5)
    experience = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
    risk_profile = calculate_risk_profile(horizon, tolerance, experience)
    st.metric("Your Risk Profile", risk_profile)

if st.button("Generate Portfolio"):
    with st.spinner("Building optimal allocation..."):
        data = fetch_data()
        if data.empty: st.stop()
        
        weights = optimize_portfolio(risk_profile, data)
        if not weights: st.stop()
        
        # Calculate metrics
        returns = data[weights.keys()].pct_change().mean().dot(list(weights.values()))
        volatility = np.sqrt(np.dot(list(weights.values()), 
                            np.dot(data[weights.keys()].pct_change().cov(), 
                                   list(weights.values())))
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual Holdings
            fig1, ax1 = plt.subplots()
            ax1.pie(weights.values(), labels=weights.keys(),
                    autopct='%1.1f%%', startangle=90)
            ax1.set_title("Individual Asset Allocation")
            st.pyplot(fig1)
            
            # Volatility Indicator
            st.metric("Annualized Volatility", f"{volatility*np.sqrt(252):.1%}")

        with col2:
            # Sector Allocation
            sector_alloc = pd.Series(weights).groupby(ASSET_UNIVERSE.get).sum()
            fig2, ax2 = plt.subplots()
            ax2.pie(sector_alloc, labels=sector_alloc.index,
                    autopct='%1.1f%%', startangle=90)
            ax2.set_title("Sector Allocation")
            st.pyplot(fig2)
            
            st.metric("Crypto Exposure", 
                      f"{sum(weights[t] for t in weights if ASSET_UNIVERSE[t] == 'Crypto'):.1%}")

        # Growth Projection
        st.subheader("€100,000 Growth Projection (5 Years)")
        growth = growth_projection(weights, data)
        
        fig3, ax3 = plt.subplots()
        ax3.plot(growth, alpha=0.1, color='grey')
        ax3.plot(pd.DataFrame(growth).quantile(0.5, axis=1), color='blue', label='Median')
        ax3.plot(pd.DataFrame(growth).quantile(0.9, axis=1), color='green', label='90th %ile')
        ax3.set_xlabel("Years")
        ax3.set_ylabel("Portfolio Value (€)")
        ax3.legend()
        st.pyplot(fig3)
