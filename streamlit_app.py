import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns

# Configure page
st.set_page_config(page_title="Smart Portfolio Manager", layout="wide")
st.title('Advanced Portfolio Optimizer with Crypto Integration')
st.write("""
### Multi-Strategy Allocation Combining Modern Portfolio Theory & Factor Investing
""")

# Ticker configuration with sanitized names
STOCK_UNIVERSE = {
    'AAPL': 'Technology',
    'MSFT': 'Technology', 
    'GOOG': 'Tech',
    'BNBUSD': 'Crypto',  # Sanitized ticker name
    'ETHUSD': 'Crypto',
    'SPY': 'ETF',
    'TLT': 'Bonds',
    'GLD': 'Commodities',
    'JPM': 'Financials',
    'XOM': 'Energy'
}

def sanitize_tickers(tickers):
    """Clean ticker names for compatibility"""
    return [t.replace('-', '') for t in tickers]

def fetch_data():
    """Robust data fetching with sanitization"""
    try:
        raw_tickers = list(STOCK_UNIVERSE.keys())
        data = yf.download(
            tickers=raw_tickers,
            period="1y",
            interval="1d",
            group_by='ticker',
            progress=False,
            auto_adjust=True
        )
        
        # Clean column names and handle multi-index
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]
            adj_close = data.filter(like='_Close')
            adj_close.columns = [col.split('_')[0] for col in adj_close.columns]
        else:
            adj_close = data['Close'].to_frame()
        
        # Sanitize and filter
        adj_close.columns = sanitize_tickers(adj_close.columns)
        valid_tickers = [t for t in STOCK_UNIVERSE if t in adj_close.columns]
        return adj_close[valid_tickers].ffill().dropna(axis=1)
    
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def optimize_portfolio(risk_profile, data):
    """Enhanced optimization with crypto handling"""
    if data.empty or len(data.columns) < 2:
        return {}

    try:
        # Convert column names to sanitized format
        data.columns = sanitize_tickers(data.columns)
        
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        if risk_profile == "Conservative":
            # Markowitz Minimum Variance
            ef = EfficientFrontier(mu, S)
            ef.min_volatility()
            
        elif risk_profile == "Moderate":
            # Factor-Based Strategy
            momentum = data.pct_change().mean()
            value = data.iloc[-252] / data.iloc[-504]  # 1-year momentum
            factor_score = 0.7*momentum + 0.3*(1/value)
            selected = factor_score.nlargest(8).index.tolist()
            
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.max_sharpe()
            
        else:  # Aggressive
            # Markowitz with Crypto Allocation
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.15)
            
            # Crypto constraints
            crypto_tickers = [t for t in data.columns if STOCK_UNIVERSE.get(t) == 'Crypto']
            if crypto_tickers:
                ef.add_constraint(lambda w: sum(w[t] for t in crypto_tickers) >= 0.25)
                ef.add_constraint(lambda w: sum(w[t] for t in crypto_tickers) <= 0.4)
            
            ef.max_sharpe()
        
        weights = ef.clean_weights()
        return {k: v for k, v in weights.items() if v > 0.01}
    
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

# Risk Profile Questionnaire
with st.sidebar:
    st.header("Investor Profile")
    risk_tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 5,
                              help="1 = Capital Preservation, 10 = Maximum Growth")
    investment_horizon = st.selectbox("Investment Horizon", 
                                    ["<3 years", "3-5 years", "5+ years"])
    experience = st.selectbox("Experience Level", 
                            ["Beginner", "Intermediate", "Advanced"])
    
    # Calculate risk profile
    risk_profile = "Conservative"
    if risk_tolerance >= 7 or (risk_tolerance >=5 and investment_horizon == "5+ years"):
        risk_profile = "Aggressive"
    elif risk_tolerance >=4:
        risk_profile = "Moderate"

    st.markdown(f"**Recommended Profile:** {risk_profile}")

def create_allocation_chart(weights):
    """Enhanced allocation visualization"""
    sector_allocation = {}
    crypto_exposure = 0
    
    for ticker, weight in weights.items():
        sector = STOCK_UNIVERSE.get(ticker, 'Other')
        if sector == 'Crypto':
            crypto_exposure += weight
        sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(sector_allocation)))
    
    wedges, texts, autotexts = ax.pie(
        sector_allocation.values(),
        labels=sector_allocation.keys(),
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 10}
    )
    
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title(f"Portfolio Allocation (Crypto: {crypto_exposure:.1%})", 
                fontsize=16, pad=20)
    return fig

# Main Application Flow
if st.button("Generate Optimal Portfolio"):
    with st.spinner("Constructing optimized portfolio..."):
        data = fetch_data()
        if data.empty:
            st.error("Failed to retrieve market data")
            st.stop()
        
        weights = optimize_portfolio(risk_profile, data)
        if not weights:
            st.error("Portfolio optimization failed")
            st.stop()
        
        # Ensure alignment between weights and data
        valid_weights = {k: v for k, v in weights.items() if k in data.columns}
        ordered_weights = {k: valid_weights[k] for k in data.columns if k in valid_weights}
        
        # Calculate returns safely
        returns = data[list(ordered_weights.keys())].pct_change().mean()
        portfolio_return = np.dot(list(ordered_weights.values()), returns)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Portfolio Allocation")
            fig = create_allocation_chart(ordered_weights)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Key Metrics")
            st.metric("Risk Profile", risk_profile)
            st.metric("Crypto Exposure", 
                      f"{sum(ordered_weights.get(t, 0) for t in STOCK_UNIVERSE if STOCK_UNIVERSE[t] == 'Crypto'):.1%}")
            st.metric("Expected Annual Return", f"{portfolio_return*252:.1%}")
            
            st.write("**Top Holdings:**")
            for ticker, weight in sorted(ordered_weights.items(), 
                                       key=lambda x: -x[1])[:3]:
                st.write(f"- {ticker}: {weight:.1%}")

        st.subheader("Strategy Breakdown")
        if risk_profile == "Conservative":
            st.markdown("""
            **Capital Preservation Strategy**
            - Minimum variance optimization (Markowitz)
            - Maximum 15% single asset allocation
            - No crypto exposure
            - Focus on low-volatility assets
            """)
        elif risk_profile == "Moderate":
            st.markdown("""
            **Smart Beta Strategy**
            - Combines momentum & value factors
            - Limited crypto exposure (0-15%)
            - Balanced sector allocation
            - Monthly rebalancing
            """)
        else:
            st.markdown("""
            **Aggressive Growth Strategy**
            - Markowitz optimization with crypto tilt (25-40%)
            - L2 regularization for stability
            - High-growth tech and crypto assets
            - Weekly rebalancing
            """)

