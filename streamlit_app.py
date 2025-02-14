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

# Asset universe with sector classification
ASSET_UNIVERSE = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOG': 'Technology',
    'SPY': 'US Equity', 'TLT': 'Treasuries', 'GLD': 'Commodities',
    'JPM': 'Financials', 'XOM': 'Energy', 'BTC-USD': 'Crypto',
    'VBK': 'Small Cap', 'VWO': 'Emerging Markets'
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
    """Advanced risk scoring system"""
    score = (
        answers['horizon'] * 0.4 +
        answers['loss_tolerance'] * 0.3 +
        answers['knowledge'] * 0.2 +
        answers['age'] * 0.1
    )
    return "Conservative" if score < 2.5 else "Moderate" if score < 4.0 else "Aggressive"

def optimize_portfolio(risk_profile, data):
    """Multi-strategy portfolio construction"""
    if data.empty: return {}
    
    try:
        mu = expected_returns.capm_return(data)
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        
        if risk_profile == "Conservative":
            ef = EfficientFrontier(None, S)
            ef.add_constraint(lambda w: w <= 0.1)
            ef.min_volatility()
            
        elif risk_profile == "Moderate":
            momentum = data.pct_change(90).mean()
            volatility = data.pct_change().std()
            factor_score = (momentum / volatility).rank(pct=True)
            selected = factor_score.nlargest(10).index
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.add_objective(objective_functions.L2_reg)
            ef.max_sharpe()
            
        else:
            market_prior = expected_returns.mean_historical_return(data)
            ef = EfficientFrontier(market_prior, S)
            ef.add_constraint(lambda w: w >= 0.05)
            ef.max_sharpe()

        return ef.clean_weights()
    
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

def calculate_metrics(weights, data):
    """Comprehensive portfolio metrics"""
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(list(weights.values()))
    
    return {
        'annual_return': portfolio_returns.mean() * 252,
        'annual_volatility': portfolio_returns.std() * np.sqrt(252),
        'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
        'max_drawdown': (portfolio_returns.cumsum().expanding().max() - 
                        portfolio_returns.cumsum()).max(),
        'beta': np.cov(portfolio_returns, returns['SPY'])[0,1] / np.var(returns['SPY']),
        'var_95': np.percentile(portfolio_returns, 5) * 100,
        'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100
    }

# Sidebar Configuration
with st.sidebar:
    st.header("Investor Profile")
    
    # Risk Profiling
    with st.expander("Risk Questionnaire", expanded=True):
        risk_answers = {
            'horizon': st.slider("Investment Horizon (Years)", 1, 10, 5),
            'loss_tolerance': st.select_slider("Max Tolerable Loss",
                                             options=["0-10%", "10-20%", "20-30%", "30%+"],
                                             value="10-20%"),
            'knowledge': st.radio("Market Knowledge", ["Novice", "Intermediate", "Expert"]),
            'age': st.number_input("Age", 18, 100, 30)
        }
        
        answer_scores = {
            'horizon': 5 - (risk_answers['horizon'] / 2),
            'loss_tolerance': {"0-10%": 1, "10-20%": 2, "20-30%": 3, "30%+": 4}[risk_answers['loss_tolerance']],
            'knowledge': {"Novice": 1, "Intermediate": 2, "Expert": 3}[risk_answers['knowledge']],
            'age': risk_answers['age'] / 25
        }
        
        risk_profile = calculate_risk_profile(answer_scores)
        st.metric("Risk Profile", risk_profile)
    
    # Investment Input
    investment = st.number_input("Investment Amount (€)", 1000, 1000000, 100000)

# Main Application
if st.button("Construct Portfolio"):
    with st.spinner("Running institutional-grade optimization..."):
        data = fetch_data()
        if data.empty: st.stop()
        
        weights = optimize_portfolio(risk_profile, data)
        if not weights: st.stop()
        
        metrics = calculate_metrics(weights, data)
        valid_weights = {k: v for k, v in weights.items() if v > 0.01}
        
        # Portfolio Composition
        with st.expander("Asset Allocation & Sector Breakdown", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.pie(valid_weights.values(), labels=valid_weights.keys(), autopct='%1.1f%%')
                ax1.set_title("Individual Asset Allocation")
                st.pyplot(fig1)
            
            with col2:
                sector_alloc = pd.Series(valid_weights).groupby(ASSET_UNIVERSE.get).sum()
                fig2, ax2 = plt.subplots()
                ax2.pie(sector_alloc, labels=sector_alloc.index, autopct='%1.1f%%')
                ax2.set_title("Sector Allocation Breakdown")
                st.pyplot(fig2)
        
        # Risk Metrics
        with st.expander("Advanced Risk Metrics", expanded=False):
            col3, col4 = st.columns(2)
            
            with col3:
                st.metric("Value at Risk (95%)", f"{metrics['var_95']:.1f}%")
                st.metric("Conditional VaR", f"{metrics['cvar_95']:.1f}%")
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
            
            with col4:
                st.metric("Annual Volatility", f"{metrics['annual_volatility']:.1%}")
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Beta vs S&P 500", f"{metrics['beta']:.2f}")
        
        # Growth Projection
        if investment > 0:
            with st.expander(f"€{investment:,.0f} Growth Projection", expanded=False):
                returns = data.pct_change().dropna().dot(list(valid_weights.values()))
                simulations = 100
                days = 252 * 5
                
                growth = investment * np.exp(np.cumsum(
                    np.random.normal(
                        returns.mean()/252, 
                        returns.std()/np.sqrt(252), 
                        (days, simulations)
                    ), 
                    axis=0
                ))
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.plot(growth, alpha=0.1, color='grey')
                ax3.plot(pd.DataFrame(growth).quantile(0.5, axis=1), color='blue', label='Median')
                ax3.set_xlabel("Trading Days")
                ax3.set_ylabel("Portfolio Value (€)")
                ax3.legend()
                st.pyplot(fig3)

