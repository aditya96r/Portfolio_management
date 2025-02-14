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

# Enhanced stock universe with reliable tickers
STOCK_UNIVERSE = [
    # US Large Caps
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B',
    # Global Blue Chips
    'NSRGY', 'TM', 'UL', 'SNY', 'AZN',
    # ETFs
    'SPY', 'IVV', 'VTI', 'VOO', 'QQQ', 'TLT', 'IEF',
    # Commodities
    'GC=F', 'SI=F', 'CL=F',
    # Bonds
    'BND', 'AGG'
]

def get_valid_tickers(tickers):
    """Advanced ticker validation with multiple checks"""
    valid = []
    invalid = []
    
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            info = ticker.info
            
            # Check multiple price indicators
            price_fields = ['regularMarketPrice', 'currentPrice', 
                           'previousClose', 'ask', 'bid']
            price = next(
                (info[field] for field in price_fields 
                if field in info and info[field] is not None),
                None
            )
            
            # Check historical data availability
            hist = ticker.history(period="7d", interval="1d")
            
            if price and not hist.empty and 'Close' in hist.columns:
                valid.append(t)
            else:
                invalid.append(t)
                
        except Exception as e:
            invalid.append(t)
    
    # Display validation results
    with st.expander("Ticker Validation Report"):
        if invalid:
            st.warning(f"⚠️ Unavailable tickers: {', '.join(invalid[:5])}{'...' if len(invalid)>5 else ''}")
        if valid:
            st.success(f"✅ Valid tickers: {', '.join(valid[:5])}{'...' if len(valid)>5 else ''}")
    
    return valid

def fetch_data():
    """Robust data fetching with multiple fallback strategies"""
    valid_tickers = get_valid_tickers(STOCK_UNIVERSE)
    if not valid_tickers:
        st.error("""
        🚨 Critical Data Issue!
        Possible solutions:
        1. Check internet connection
        2. Try different assets
        3. Retry during market hours (9:30 AM - 4 PM EST)
        """)
        return pd.DataFrame()
    
    # Try different time periods
    for period in ["2y", "1y", "6mo"]:
        try:
            data = yf.download(
                tickers=valid_tickers,
                period=period,
                interval="1d",
                group_by='ticker',
                progress=False,
                auto_adjust=True,
                threads=True
            )
            
            # Handle data structure variations
            if len(valid_tickers) == 1:
                adj_close = data[['Close']].rename(columns={'Close': valid_tickers[0]})
            else:
                adj_close = data.xs('Close', level=1, axis=1, drop_level=False)
            
            cleaned_data = adj_close.ffill().dropna(axis=1)
            if not cleaned_data.empty:
                return cleaned_data
            
        except Exception as e:
            continue
    
    st.error("🔴 Data retrieval failed across all attempts")
    return pd.DataFrame()

def calculate_momentum(data):
    """Momentum calculation with validation"""
    if data.empty or len(data) < 20:
        return pd.Series(dtype=float)
    return data.pct_change().dropna().last('3M').mean()

def calculate_value(data):
    """Value factor calculation with multiple PE sources"""
    pe_ratios = {}
    for ticker in data.columns:
        try:
            info = yf.Ticker(ticker).info
            pe = info.get('trailingPE') or info.get('forwardPE') or np.nan
            pe_ratios[ticker] = pe if pd.notnull(pe) else np.nan
        except:
            pe_ratios[ticker] = np.nan
    return pd.Series(pe_ratios).replace([np.inf, -np.inf], np.nan).dropna()

def optimize_portfolio(risk_category, data):
    """Portfolio optimization with multiple fallbacks"""
    if data.empty:
        return {}
    
    # Minimum asset check
    if len(data.columns) < 2:
        st.warning("⚠️ Minimum 2 assets required for optimization")
        return {data.columns[0]: 1.0} if len(data.columns) == 1 else {}

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
            
            # Fallback to equal weight if factors missing
            if momentum.empty or value.empty:
                return {t: 1/len(data.columns) for t in data.columns}
                
            momentum_norm = momentum.rank(pct=True)
            value_norm = (1/value.rank(pct=True)).fillna(0)
            combined_score = 0.5*momentum_norm + 0.5*value_norm
            
            selected = combined_score.nlargest(10).index.tolist()
            selected = selected if len(selected) >=2 else data.columns.tolist()[:2]
                
            ef = EfficientFrontier(returns[selected], cov_matrix.loc[selected, selected])
            ef.max_sharpe()
            weights = ef.clean_weights()
            
        else:  # Conservative
            volatility = data.pct_change().std().nsmallest(5)
            weights = {t: 1/len(volatility) for t in volatility.index}
            
        return {k: v for k, v in weights.items() if v > 0.01}
        
    except Exception as e:
        st.error(f"Optimization Error: {str(e)}")
        return {}

# Risk Profile Sidebar
with st.sidebar:
    st.header("Investor Profile")
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=range(1, 11),
        value=5,
        help="1 = Capital Preservation, 10 = Maximum Growth"
    )
    investment_horizon = st.radio(
        "Investment Horizon",
        ["Short-term (<3 years)", "Medium-term (3-5 years)", "Long-term (>5 years)"]
    )
    st.markdown("---")
    st.write("💡 Tip: Higher risk tolerance requires longer time horizons")

# Main Application
if st.button("Generate Optimal Portfolio 🚀"):
    with st.spinner("Analyzing global markets..."):
        data = fetch_data()
        
        if data.empty:
            st.stop()
        
        # Determine risk profile
        if risk_tolerance <= 3:
            risk_category = "Conservative"
            allocation = {"Stocks": 30, "Bonds": 50, "Commodities": 15, "Cash": 5}
        elif risk_tolerance <= 7:
            risk_category = "Moderate"
            allocation = {"Stocks": 60, "Bonds": 25, "Commodities": 10, "Cash": 5}
        else:
            risk_category = "Aggressive"
            allocation = {"Stocks": 90, "Bonds": 5, "Commodities": 0, "Cash": 5}

        # Portfolio Optimization
        stock_weights = optimize_portfolio(risk_category, data)
        if not stock_weights:
            st.error("Portfolio Optimization Failed")
            st.stop()
        
        # Create Allocation
        total_stock_weight = sum(stock_weights.values())
        full_allocation = {}
        
        # Stock Allocation
        for stock, weight in stock_weights.items():
            full_allocation[stock] = (weight / total_stock_weight) * allocation["Stocks"]
        
        # Other Assets
        for asset, weight in allocation.items():
            if asset != "Stocks":
                full_allocation[asset] = weight

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(full_allocation)))
        
        wedges, texts, autotexts = ax.pie(
            full_allocation.values(),
            labels=full_allocation.keys(),
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            textprops={'fontsize': 8}
        )
        
        plt.setp(autotexts, size=8, weight="bold")
        ax.set_title(f"{risk_category} Portfolio Allocation", fontsize=16, pad=20)
        
        # Display
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(fig)
        with col2:
            st.subheader("Portfolio Summary")
            st.metric("Risk Category", risk_category)
            st.metric("Total Assets", f"{len(full_allocation)} holdings")
            
            # Performance Metrics
            returns = data.pct_change().mean().dot(list(stock_weights.values()))
            volatility = np.sqrt(
                np.dot(list(stock_weights.values()), 
                      np.dot(data.pct_change().cov(), 
                            list(stock_weights.values())))
            )
            st.metric("Expected Annual Return", f"{returns*252:.1%}")
            st.metric("Expected Volatility", f"{volatility*np.sqrt(252):.1%}")
            st.metric("Risk-Adjusted Return", f"{returns/volatility:.2f}")
        
        # Detailed Breakdown
        st.subheader("Detailed Allocation")
        alloc_df = pd.DataFrame.from_dict(full_allocation, 
                                        orient='index',
                                        columns=['Allocation (%)'])
        st.dataframe(
            alloc_df.sort_values(by='Allocation (%)', ascending=False),
            height=400,
            use_container_width=True
        )

