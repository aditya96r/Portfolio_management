import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns

# Configure page
st.set_page_config(page_title="Advanced Portfolio Optimizer", layout="wide")
st.title('Professional Portfolio Manager')
st.write("""
### Institutional-Grade Asset Allocation with Risk Management
""")

ASSET_UNIVERSE = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOG': 'Tech',
    'SPY': 'Equity', 'TLT': 'Bonds', 'GLD': 'Commodities',
    'JPM': 'Financials', 'XOM': 'Energy', 'ARKK': 'Innovation',
    'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto'
}

def fetch_data():
    """Robust data fetcher with validation"""
    try:
        data = yf.download(
            list(ASSET_UNIVERSE.keys()),
            period="3y",
            interval="1d",
            group_by='ticker',
            progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            return data.xs('Close', level=1, axis=1).ffill().dropna(axis=1)
        return data['Close'].ffill().dropna(axis=1)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_risk_profile(answers):
    """Improved aggressive scoring system"""
    score = (
        (5 - (answers['horizon']/2)) * 0.3 +  # Reduced horizon impact
        {"0-10%": 1, "10-20%": 2, "20-30%": 3, "30%+": 4}[answers['loss_tolerance']] * 0.5 +  # Increased weight
        {"Novice": 1, "Intermediate": 2, "Expert": 3}[answers['knowledge']] * 0.1 +
        (answers['age'] < 40) * 2  # Strong age impact
    )
    if score < 3.5: return "Conservative"
    elif score < 5.5: return "Moderate"
    else: return "Aggressive"

def optimize_portfolio(risk_profile, data):
    """Error-proof portfolio construction"""
    if data.empty or len(data.columns) < 3:
        return {}
    
    try:
        mu = expected_returns.capm_return(data)
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        
        if risk_profile == "Conservative":
            ef = EfficientFrontier(None, S)
            ef.add_constraint(lambda w: w <= 0.15)
            ef.min_volatility()
            
        elif risk_profile == "Moderate":
            momentum = data.pct_change(90).mean()
            volatility = data.pct_change().std()
            selected = (momentum / volatility).nlargest(8).index.tolist()
            
            # Create new EF instance for selected assets
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.add_objective(objective_functions.L2_reg)
            ef.max_sharpe()
            
        else:  # Aggressive
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.05)  # Reduced regularization
            crypto_assets = [t for t in data.columns if ASSET_UNIVERSE[t] == 'Crypto']
            if crypto_assets:
                ef.add_constraint(lambda w: sum(w[c] for c in crypto_assets) >= 0.35)
            ef.max_sharpe()

        return ef.clean_weights()
    
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

def calculate_metrics(weights, data):
    """Robust metrics calculation"""
    try:
        returns = data.pct_change().dropna()
        valid_assets = [a for a in weights if a in returns.columns]
        aligned_weights = np.array([weights[a] for a in valid_assets])
        
        if len(valid_assets) == 0:
            return {}
            
        portfolio_returns = returns[valid_assets].dot(aligned_weights)
        
        return {
            'annual_return': portfolio_returns.mean() * 252,
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
            'max_drawdown': (portfolio_returns.cumsum().expanding().max() - portfolio_returns.cumsum()).max(),
            'var_95': np.percentile(portfolio_returns, 5) * 100,
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100
        }
    except Exception as e:
        st.error(f"Metrics error: {str(e)}")
        return {}

# Sidebar Configuration
with st.sidebar:
    st.header("Investor Profile")
    with st.expander("Risk Assessment", expanded=True):
        risk_answers = {
            'horizon': st.slider("Investment Horizon (Years)", 1, 10, 3),
            'loss_tolerance': st.select_slider("Max Loss Tolerance", 
                                             options=["0-10%", "10-20%", "20-30%", "30%+"],
                                             value="20-30%"),
            'knowledge': st.radio("Market Experience", ["Novice", "Intermediate", "Expert"]),
            'age': st.number_input("Age", 18, 100, 35)
        }
        risk_profile = calculate_risk_profile(risk_answers)
        st.metric("Risk Profile", risk_profile)
    
    investment = st.number_input("Investment Amount (€)", 1000, 1000000, 100000)

# Main Application
if st.button("Generate Portfolio"):
    with st.spinner("Constructing optimal allocation..."):
        data = fetch_data()
        if data.empty: st.stop()
        
        weights = optimize_portfolio(risk_profile, data)
        if not weights: st.stop()
        
        valid_weights = {k: v for k, v in weights.items() if v > 0.01}
        metrics = calculate_metrics(valid_weights, data)
        
        # Portfolio Composition
        with st.expander("Asset Allocation", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.pie(valid_weights.values(), labels=valid_weights.keys(), autopct='%1.1f%%')
                ax.set_title("Individual Holdings")
                st.pyplot(fig)
            with col2:
                sector_alloc = pd.Series(valid_weights).groupby(ASSET_UNIVERSE.get).sum()
                fig, ax = plt.subplots()
                ax.pie(sector_alloc, labels=sector_alloc.index, autopct='%1.1f%%')
                ax.set_title("Sector Allocation")
                st.pyplot(fig)
        
        # Risk Metrics
        with st.expander("Risk Analysis", expanded=False):
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Expected Annual Return", f"{metrics.get('annual_return', 0):.1%}")
                st.metric("Annual Volatility", f"{metrics.get('annual_volatility', 0):.1%}")
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            with col4:
                st.metric("Value at Risk (95%)", f"{metrics.get('var_95', 0):.1f}%")
                st.metric("Conditional VaR", f"{metrics.get('cvar_95', 0):.1f}%")
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}")
        
        # Growth Projection
        if investment > 0 and metrics.get('annual_return'):
            with st.expander(f"Growth Projection - €{investment:,.0f}", expanded=False):
                years = st.select_slider("Projection Period", [5, 10], 5)
                simulations = 500
                daily_returns = np.random.normal(
                    metrics['annual_return']/252,
                    metrics['annual_volatility']/np.sqrt(252),
                    (252*years, simulations)
                )
                growth = investment * np.exp(np.cumsum(daily_returns, axis=0))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(growth, alpha=0.1, color='steelblue')
                ax.plot(pd.DataFrame(growth).median(axis=1), color='darkorange', linewidth=2, label='Median')
                ax.set_title(f"{years}-Year Growth Potential")
                ax.set_xlabel("Trading Days")
                ax.set_ylabel("Portfolio Value (€)")
                ax.legend()
                st.pyplot(fig)

