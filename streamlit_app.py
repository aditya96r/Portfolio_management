import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns

# Configure page
st.set_page_config(page_title="Professional Portfolio Manager", layout="wide")
st.title('Institutional Portfolio Optimizer')
st.write("""
### Multi-Strategy Allocation with Risk Management
""")

ASSET_UNIVERSE = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOG': 'Technology',
    'SPY': 'US Equity', 'TLT': 'Treasuries', 'GLD': 'Commodities',
    'JPM': 'Financials', 'XOM': 'Energy', 'VBK': 'Small Cap', 'VWO': 'Emerging Markets'
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
    """Improved risk scoring system"""
    score = (
        (5 - (answers['horizon'] / 2)) * 0.4 +  # Horizon impact reduced
        {"0-10%": 1, "10-20%": 2, "20-30%": 3, "30%+": 4}[answers['loss_tolerance']] * 0.4 +
        {"Novice": 1, "Intermediate": 2, "Expert": 3}[answers['knowledge']] * 0.2 +
        (answers['age'] / 40)  # Age impact reduced
    )
    if score < 3.0: return "Conservative"
    elif score < 4.5: return "Moderate"
    else: return "Aggressive"

def optimize_portfolio(risk_profile, data):
    """Robust portfolio construction"""
    if data.empty or len(data.columns) < 3:
        return {}
    
    try:
        mu = expected_returns.capm_return(data)
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        ef = EfficientFrontier(mu, S)

        if risk_profile == "Conservative":
            ef.min_volatility()
            ef.add_constraint(lambda w: w <= 0.15)
        elif risk_profile == "Moderate":
            selected = data.columns[(data.pct_change(90).mean() / data.pct_change().std()).nlargest(8).index]
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.max_sharpe()
        else:
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.max_sharpe()

        return ef.clean_weights()
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

def calculate_metrics(weights, data):
    """Error-resistant metrics calculation"""
    try:
        returns = data.pct_change().dropna()
        aligned_weights = pd.Series(weights).reindex(returns.columns, fill_value=0)
        portfolio_returns = returns.dot(aligned_weights)
        
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
    with st.expander("Risk Questionnaire", expanded=True):
        risk_answers = {
            'horizon': st.slider("Investment Horizon (Years)", 1, 10, 5),
            'loss_tolerance': st.select_slider("Max Tolerable Loss", ["0-10%", "10-20%", "20-30%", "30%+"], "10-20%"),
            'knowledge': st.radio("Market Knowledge", ["Novice", "Intermediate", "Expert"]),
            'age': st.number_input("Age", 18, 100, 30)
        }
        risk_profile = calculate_risk_profile(risk_answers)
        st.metric("Your Risk Profile", risk_profile)
    
    investment = st.number_input("Investment Amount (€)", 1000, 1000000, 100000)

# Main Application
if st.button("Construct Portfolio"):
    with st.spinner("Running institutional-grade optimization..."):
        data = fetch_data()
        if data.empty: st.stop()
        
        weights = optimize_portfolio(risk_profile, data)
        if not weights: st.stop()
        
        valid_weights = {k: v for k, v in weights.items() if v > 0.01}
        metrics = calculate_metrics(valid_weights, data)
        
        # Portfolio Composition
        with st.expander("Asset Allocation & Sector Breakdown", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.pie(valid_weights.values(), labels=valid_weights.keys(), autopct='%1.1f%%')
                ax.set_title("Asset Allocation")
                st.pyplot(fig)
            with col2:
                sector_alloc = pd.Series(valid_weights).groupby(ASSET_UNIVERSE.get).sum()
                fig, ax = plt.subplots()
                ax.pie(sector_alloc, labels=sector_alloc.index, autopct='%1.1f%%')
                ax.set_title("Sector Allocation")
                st.pyplot(fig)
        
        # Risk Metrics
        with st.expander("Advanced Risk Metrics", expanded=False):
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Value at Risk (95%)", f"{metrics.get('var_95', 0):.1f}%")
                st.metric("Conditional VaR", f"{metrics.get('cvar_95', 0):.1f}%")
            with col4:
                st.metric("Annual Volatility", f"{metrics.get('annual_volatility', 0):.1%}")
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        # Growth Projection
        if investment > 0 and metrics.get('annual_return'):
            with st.expander(f"€{investment:,.0f} Growth Projection", expanded=False):
                years = st.selectbox("Projection Period", [5, 10], index=0)
                simulations = 500
                daily_returns = np.random.normal(
                    metrics['annual_return']/252,
                    metrics['annual_volatility']/np.sqrt(252),
                    (252*years, simulations)
                )
                growth = investment * np.exp(np.cumsum(daily_returns, axis=0))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(growth, alpha=0.1, color='grey')
                ax.plot(pd.DataFrame(growth).median(axis=1), color='blue', label='Median')
                ax.set_title(f"{years}-Year Growth Projection")
                ax.legend()
                st.pyplot(fig)

# requirements.txt
"""
numpy==1.26.4
pandas==2.2.2
yfinance==0.2.52
matplotlib==3.8.3
streamlit==1.35.0
pypfopt==1.5.5
"""
