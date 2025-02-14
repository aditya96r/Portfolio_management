import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns
from pypfopt import plotting

# Configure page
st.set_page_config(page_title="Professional Portfolio Manager", layout="wide")
st.title('Institutional-Grade Portfolio Optimizer')
st.write("""
### Multi-Strategy Allocation Framework with Risk Management
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
    weights = {
        'horizon': 0.4,
        'loss_tolerance': 0.3,
        'knowledge': 0.2,
        'age': 0.1
    }
    
    score = (
        weights['horizon'] * answers['horizon'] +
        weights['loss_tolerance'] * answers['loss_tolerance'] +
        weights['knowledge'] * answers['knowledge'] +
        weights['age'] * answers['age']
    )
    
    if score < 2.5: return "Conservative"
    elif score < 4.0: return "Moderate"
    else: return "Aggressive"

def optimize_portfolio(risk_profile, data):
    """Multi-strategy portfolio construction"""
    if data.empty: return {}
    
    try:
        mu = expected_returns.capm_return(data)
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        
        if risk_profile == "Conservative":
            # Minimum Variance Portfolio (Markowitz)
            ef = EfficientFrontier(None, S)
            ef.add_constraint(lambda w: w <= 0.1)
            ef.min_volatility()
            
        elif risk_profile == "Moderate":
            # Factor-Based Portfolio
            momentum = data.pct_change(90).mean()
            volatility = data.pct_change().std()
            factor_score = (momentum / volatility).rank(pct=True)
            selected = factor_score.nlargest(10).index
            
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.add_objective(objective_functions.L2_reg)
            ef.max_sharpe()
            
        else:  # Aggressive
            # Black-Litterman Model with Views
            market_prior = expected_returns.mean_historical_return(data)
            bl = BlackLittermanModel(S, pi=market_prior)
            bl_views = {
                'AAPL': 0.15,  # Bullish on Apple
                'TLT': -0.05    # Bearish on Bonds
            }
            bl.add_views(bl_views)
            ret_bl = bl.bl_returns()
            
            ef = EfficientFrontier(ret_bl, S)
            ef.add_constraint(lambda w: w >= 0.05)  # No shorting
            ef.max_sharpe()

        weights = ef.clean_weights()
        return {k: v for k, v in weights.items() if v > 0.01}
    
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
        'beta': np.cov(portfolio_returns, returns['SPY'])[0,1] / np.var(returns['SPY'])
    }

# Risk Profiling Questionnaire
with st.sidebar.expander("Risk Profile Assessment", expanded=True):
    st.subheader("Investor Profile Questionnaire")
    
    risk_answers = {
        'horizon': st.slider("Investment Horizon (Years)", 1, 10, 5,
                            help="Expected time until funds are needed"),
        'loss_tolerance': st.select_slider("Maximum Tolerable Loss",
                                         options=["0-10%", "10-20%", "20-30%", "30%+"],
                                         value="10-20%"),
        'knowledge': st.radio("Market Knowledge",
                             ["Novice", "Intermediate", "Expert"]),
        'age': st.number_input("Age", 18, 100, 30)
    }
    
    # Convert qualitative answers to scores
    answer_scores = {
        'horizon': 5 - (risk_answers['horizon'] / 2),  # Shorter horizon = more conservative
        'loss_tolerance': {"0-10%": 1, "10-20%": 2, "20-30%": 3, "30%+": 4}[risk_answers['loss_tolerance']],
        'knowledge': {"Novice": 1, "Intermediate": 2, "Expert": 3}[risk_answers['knowledge']],
        'age': risk_answers['age'] / 25  # Younger = more aggressive
    }
    
    risk_profile = calculate_risk_profile(answer_scores)
    st.metric("Your Risk Profile", risk_profile)

# Main Application
if st.button("Construct Optimal Portfolio"):
    with st.spinner("Running institutional-grade optimization..."):
        data = fetch_data()
        if data.empty: st.stop()
        
        weights = optimize_portfolio(risk_profile, data)
        if not weights: st.stop()
        
        metrics = calculate_metrics(weights, data)
        
        # Portfolio Monitoring Dashboard
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Core Metrics")
            st.metric("Expected Annual Return", f"{metrics['annual_return']:.1%}")
            st.metric("Annual Volatility", f"{metrics['annual_volatility']:.1%}")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            
        with col2:
            st.subheader("Risk Metrics")
            st.metric("Maximum Drawdown", f"{metrics['max_drawdown']:.1%}")
            st.metric("Beta vs S&P 500", f"{metrics['beta']:.2f}")
            st.metric("VaR (95%)", f"{np.percentile(portfolio_returns, 5)*100:.1f}%")
            
        with col3:
            st.subheader("Allocation")
            fig1, ax1 = plt.subplots()
            plotting.plot_weights(weights, ax=ax1)
            st.pyplot(fig1)
        
        # Efficient Frontier Visualization
        st.subheader("Efficient Frontier Analysis")
        fig2, ax2 = plt.subplots()
        plotting.plot_efficient_frontier(
            EfficientFrontier(
                expected_returns.mean_historical_return(data),
                risk_models.sample_cov(data)
            ),
            ax=ax2
        )
        st.pyplot(fig2)
        
        # Historical Performance
        st.subheader("Historical Risk/Reward Profile")
        fig3, ax3 = plt.subplots()
        ax3.scatter(metrics['annual_volatility'], metrics['annual_return'],
                    c=metrics['sharpe_ratio'], s=100, cmap='viridis')
        ax3.set_xlabel("Volatility")
        ax3.set_ylabel("Return")
        plt.colorbar(ax3.collections[0], label='Sharpe Ratio')
        st.pyplot(fig3)

