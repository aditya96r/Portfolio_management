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
### Dynamic Portfolio Optimization using Modern Finance Theory
""")

# Updated stock universe with verified tickers
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA',
    'JNJ', 'PFE', 'MRK', 'JPM', 'BAC',
    'GS', 'WMT', 'TGT', 'COST', 'XOM',
    'CVX', 'UNH', 'PG', 'DIS', 'NKE'
]

def get_valid_tickers(tickers):
    """Robust validation of tickers using multiple checks"""
    valid = []
    for t in tickers:
        try:
            # Check basic info
            info = yf.Ticker(t).info
            if not info.get('regularMarketPrice'):
                continue
                
            # Check historical data
            hist = yf.Ticker(t).history(period="7d")
            if hist.empty or 'Close' not in hist.columns:
                continue
                
            valid.append(t)
        except Exception as e:
            continue
    return valid

def fetch_data():
    """Fetch and validate stock data with robust error handling"""
    valid_tickers = get_valid_tickers(STOCK_UNIVERSE)
    if not valid_tickers:
        st.error("🚨 No valid tickers found! Check your stock symbols.")
        return pd.DataFrame()
    
    try:
        data = yf.download(
            tickers=valid_tickers,
            period="2y",
            interval="1d",
            group_by='ticker',
            progress=False,
            auto_adjust=True
        )
        
        # Handle different data structures
        if len(valid_tickers) == 1:
            adj_close = data[['Close']].rename(columns={'Close': valid_tickers[0]})
        else:
            adj_close = data.xs('Close', level=1, axis=1, drop_level=False)
            
        # Clean and validate data
        adj_close = adj_close.ffill().dropna(axis=1)
        if adj_close.empty:
            st.warning("⚠️ Data available but contains missing values")
            return pd.DataFrame()
            
        return adj_close
    
    except Exception as e:
        st.error(f"🔴 Data retrieval failed: {str(e)}")
        return pd.DataFrame()

def calculate_momentum(data):
    """Calculate momentum factor with validation"""
    if data.empty:
        return pd.Series()
    returns = data.pct_change().dropna()
    return returns.last('3M').mean()

def calculate_value(data):
    """Calculate value factor with enhanced error handling"""
    pe_ratios = {}
    for ticker in data.columns:
        try:
            info = yf.Ticker(ticker).info
            pe = info.get('trailingPE', info.get('forwardPE', np.nan))
            pe_ratios[ticker] = pe if pd.notnull(pe) else np.nan
        except:
            pe_ratios[ticker] = np.nan
    return pd.Series(pe_ratios).replace([np.inf, -np.inf], np.nan).dropna()

def optimize_portfolio(risk_category, data):
    """Robust portfolio optimization with fallbacks"""
    if data.empty or len(data.columns) < 2:
        st.error("❌ Insufficient data for optimization")
        return {}
    
    try:
        returns = expected_returns.mean_historical_return(data)
        cov_matrix = risk_models.sample_cov(data)
        
        if risk_category == "Aggressive":
            ef = EfficientFrontier(returns, cov_matrix)
            ef.add_objective(objective_functions.L2_reg)
            ef.max_sharpe()
            weights = ef.clean_weights()
            
        elif risk_category == "Moderate":
            momentum = calculate_momentum(data)
            value = calculate_value(data)
            
            if momentum.empty or value.empty:
                st.warning("⚠️ Factor data incomplete, using equal weighting")
                return {t: 1/len(data.columns) for t in data.columns}
                
            momentum_norm = momentum.rank(pct=True)
            value_norm = (1/value.rank(pct=True)).fillna(0)
            combined_score = 0.5*momentum_norm + 0.5*value_norm
            
            selected = combined_score.nlargest(10).index.tolist()
            if len(selected) < 2:
                selected = data.columns.tolist()[:2]
                
            ef = EfficientFrontier(returns[selected], cov_matrix.loc[selected, selected])
            ef.max_sharpe()
            weights = ef.clean_weights()
            
        else:  # Conservative
            volatility = data.pct_change().std().nsmallest(5)
            weights = {t: 1/len(volatility) for t in volatility.index}
            
        return {k: v for k, v in weights.items() if v > 0.01}
        
    except Exception as e:
        st.error(f"🔴 Optimization failed: {str(e)}")
        return {}

# Risk Assessment Sidebar
with st.sidebar:
    st.header("Investor Profile")
    risk_tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 5,
                              help="1 = Very Conservative, 10 = Very Aggressive")
    investment_horizon = st.selectbox("Investment Horizon", 
        ["1-3 years", "3-5 years", "5+ years"])
    experience = st.selectbox("Experience Level", 
        ["Beginner", "Intermediate", "Advanced"])

# Main application logic
if st.button("Build Optimal Portfolio 🔄"):
    with st.spinner("Analyzing market data..."):
        data = fetch_data()
        
        if data.empty:
            st.stop()
            
        # Determine risk category
        if risk_tolerance <= 3:
            risk_category = "Conservative"
            allocation = {"Stocks": 30, "Bonds": 50, "REITs": 15, "Cash": 5}
        elif risk_tolerance <= 7:
            risk_category = "Moderate"
            allocation = {"Stocks": 60, "Bonds": 25, "REITs": 10, "Cash": 5}
        else:
            risk_category = "Aggressive"
            allocation = {"Stocks": 90, "Bonds": 5, "REITs": 0, "Cash": 5}

        # Optimize stock portfolio
        stock_weights = optimize_portfolio(risk_category, data)
        if not stock_weights:
            st.error("❌ Failed to optimize portfolio")
            st.stop()
            
        # Create full allocation
        full_allocation = {}
        stock_percentage = allocation["Stocks"] / sum(stock_weights.values())
        for stock, weight in stock_weights.items():
            full_allocation[stock] = weight * stock_percentage
            
        for asset, weight in allocation.items():
            if asset != "Stocks":
                full_allocation[asset] = weight

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(full_allocation.values(), labels=full_allocation.keys(),
               autopct='%1.1f%%', startangle=90,
               colors=plt.cm.tab20.colors,
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        ax.set_title(f"{risk_category} Portfolio Allocation", pad=20, fontsize=16)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(fig)
        with col2:
            st.subheader("Portfolio Strategy")
            st.markdown(f"""
            - **Risk Profile:** {risk_category}
            - **Optimization Method:** {'Factor Investing' if risk_category == 'Moderate' else 'Markowitz'}
            - **Asset Classes:** {len(full_allocation)} holdings
            - **Rebalancing:** {'Quarterly' if risk_category == 'Conservative' else 'Monthly'}
            """)
            
            returns = data.pct_change().mean().dot(list(stock_weights.values()))
            volatility = np.sqrt(np.dot(list(stock_weights.values()), 
                                     np.dot(data.pct_change().cov(), 
                                            list(stock_weights.values()))))
            st.metric("Expected Annual Return", f"{returns*252:.1%}")
            st.metric("Expected Volatility", f"{volatility*np.sqrt(252):.1%}")
            st.metric("Sharpe Ratio", f"{returns/volatility:.2f}")

        st.subheader("Detailed Allocation")
        allocation_df = pd.DataFrame.from_dict(full_allocation, 
                                              orient='index',
                                              columns=['Allocation (%)'])
        st.dataframe(allocation_df.sort_values(by='Allocation (%)', ascending=False),
                    height=300)
