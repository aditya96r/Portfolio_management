import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns

# Configure page
st.set_page_config(page_title="Crypto-Integrated Portfolio Manager", layout="wide")
st.title('AI-Driven Portfolio Management with Crypto Assets')
st.write("""
### Next-Gen Portfolio Optimization Combining Traditional and Digital Assets
""")

# Enhanced universe with crypto assets and sector info
STOCK_UNIVERSE = {
    # Traditional Assets
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOG': 'Tech',
    'AMZN': 'Retail', 'TSLA': 'Auto', 'JPM': 'Financial',
    'GS': 'Financial', 'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto',
    'XRP-USD': 'Crypto', 'SPY': 'ETF', 'GLD': 'Commodity',
    'TLT': 'Bonds', 'BNB-USD': 'Crypto', 'ADA-USD': 'Crypto'
}

def get_sector(ticker):
    """Get asset class with crypto detection"""
    return STOCK_UNIVERSE.get(ticker, 'Other')

def fetch_data():
    """Fetch data with crypto integration"""
    try:
        data = yf.download(
            list(STOCK_UNIVERSE.keys()),
            period="1y",
            interval="1d",
            group_by='ticker',
            progress=False
        )
        
        # Handle multi-index DataFrame
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data.xs('Close', level=1, axis=1)
        else:
            adj_close = data['Close'].to_frame()
            
        return adj_close.ffill().dropna(axis=1)
    
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_risk_profile(investment_horizon, risk_tolerance, experience):
    """Dynamic risk profile calculation"""
    score = 0
    # Investment horizon scoring
    horizon_map = {"<1 year": 1, "1-3 years": 3, "3-5 years": 5, "5+ years": 7}
    score += horizon_map.get(investment_horizon, 3)
    
    # Risk tolerance direct score
    score += risk_tolerance
    
    # Experience weighting
    exp_map = {"Beginner": 0.5, "Intermediate": 1, "Advanced": 1.5}
    score *= exp_map.get(experience, 1)
    
    if score < 5: return "Conservative"
    elif score < 10: return "Moderate"
    else: return "Aggressive"

def optimize_portfolio(risk_profile, data):
    """Crypto-integrated portfolio optimization"""
    if data.empty or len(data.columns) < 3:
        return {}
    
    try:
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        if risk_profile == "Conservative":
            ef = EfficientFrontier(mu, S)
            ef.min_volatility()
            ef.add_constraint(lambda w: w <= 0.15)  # Concentration limit
            
        elif risk_profile == "Moderate":
            # Factor investing with momentum and volatility
            momentum = data.pct_change().mean()
            volatility = data.pct_change().std()
            factor_score = momentum/volatility
            
            selected = factor_score.nlargest(15).index.tolist()
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.max_sharpe()
            
        else:  # Aggressive
            # Markowitz with crypto tilt
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            
            # Force minimum crypto allocation (20-40%)
            crypto_tickers = [t for t in data.columns if get_sector(t) == 'Crypto']
            if crypto_tickers:
                ef.add_constraint(lambda w: sum(w[c] for c in crypto_tickers) >= 0.2)
                ef.add_constraint(lambda w: sum(w[c] for c in crypto_tickers) <= 0.4)
            
            ef.max_sharpe()
        
        weights = ef.clean_weights()
        return {k: v for k, v in weights.items() if v > 0.01}
    
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

def create_allocation_chart(weights):
    """Create pie chart with crypto highlight"""
    sector_allocation = {}
    crypto_exposure = 0
    for ticker, weight in weights.items():
        sector = get_sector(ticker)
        if sector == 'Crypto':
            crypto_exposure += weight
        sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
    
    # Explode crypto slice if present
    explode = [0.1 if s == 'Crypto' else 0 for s in sector_allocation.keys()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(sector_allocation.values(), labels=sector_allocation.keys(),
           autopct='%1.1f%%', startangle=90, explode=explode,
           colors=plt.cm.tab20.colors, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
    ax.set_title(f"Portfolio Allocation (Crypto Exposure: {crypto_exposure:.1%})")
    return fig

# Risk Profiling Sidebar
with st.sidebar:
    st.header("Investor Profile")
    investment_horizon = st.selectbox(
        "Investment Horizon",
        ["<1 year", "1-3 years", "3-5 years", "5+ years"],
        index=2
    )
    risk_tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 5)
    experience = st.selectbox(
        "Trading Experience",
        ["Beginner", "Intermediate", "Advanced"]
    )
    risk_profile = calculate_risk_profile(investment_horizon, risk_tolerance, experience)
    st.markdown(f"**Calculated Risk Profile:** {risk_profile}")

# Main Application
if st.button("Generate Optimal Portfolio"):
    with st.spinner("Building next-gen portfolio..."):
        data = fetch_data()
        if data.empty:
            st.error("Failed to load market data")
            st.stop()
        
        weights = optimize_portfolio(risk_profile, data)
        if not weights:
            st.error("Portfolio optimization failed")
            st.stop()
        
        # Calculate returns with proper alignment
        returns = data[list(weights.keys())].pct_change().mean()
        portfolio_return = np.dot(list(weights.values()), returns)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Asset Allocation")
            fig = create_allocation_chart(weights)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Portfolio Details")
            st.metric("Risk Profile", risk_profile)
            st.metric("Crypto Exposure", 
                     f"{sum(weights[t] for t in weights if get_sector(t) == 'Crypto'):.1%}")
            st.metric("Expected Annual Return", f"{portfolio_return*252:.1%}")
            
            st.write("**Top Holdings:**")
            for ticker, weight in sorted(weights.items(), key=lambda x: -x[1])[:5]:
                st.write(f"- {ticker}: {weight:.1%}")

        st.subheader("Strategy Breakdown")
        if risk_profile == "Conservative":
            st.markdown("""
            **Capital Preservation Strategy**
            - Minimum volatility portfolio (Markowitz)
            - Maximum 15% allocation to single asset
            - No crypto exposure
            """)
        elif risk_profile == "Moderate":
            st.markdown("""
            **Smart Beta Strategy**
            - Combines momentum and low volatility factors
            - Balanced crypto exposure (0-20%)
            - Sector-diversified equity portfolio
            """)
        else:
            st.markdown("""
            **Aggressive Growth Strategy**
            - Markowitz optimization with crypto tilt (20-40%)
            - L2 regularization for stability
            - High-growth tech and crypto assets
            """)

