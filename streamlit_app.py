import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, objective_functions, risk_models, expected_returns
import openai

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "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")

# Configure Streamlit page
st.set_page_config(page_title="Professional Portfolio Manager", layout="wide")
st.title('Advanced Wealth Optimizer')
st.write("### Institutional-Grade Portfolio Construction with Risk Management")

# Asset universe with valid crypto symbols
ASSET_UNIVERSE = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOG': 'Tech',
    'SPY': 'Equity', 'TLT': 'Bonds', 'GLD': 'Commodities',
    'JPM': 'Financials', 'XOM': 'Energy', 'ARKK': 'Innovation',
    'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto'  # Corrected symbols
}

def sanitize_tickers(tickers):
    return [t.replace('-', '') for t in tickers]

def fetch_data():
    try:
        data = yf.download(
            sanitize_tickers(ASSET_UNIVERSE.keys()),
            period="3y",
            interval="1d",
            group_by='ticker',
            progress=False,
            auto_adjust=True
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]
            close_prices = data.filter(like='_Close')
            close_prices.columns = [col.split('_')[0] for col in close_prices.columns]
        else:
            close_prices = data['Close'].to_frame()
        close_prices.columns = sanitize_tickers(close_prices.columns)
        return close_prices[[t for t in ASSET_UNIVERSE if t in close_prices.columns]].ffill().dropna(axis=1)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_risk_profile(answers):
    horizon_score = (answers['horizon'] / 1.5) * 0.4
    loss_score = {"0-10%": 1, "10-20%": 3, "20-30%": 5, "30%+": 7}[answers['loss_tolerance']] * 0.6
    knowledge_score = {"Novice": 1, "Intermediate": 3, "Expert": 5}[answers['knowledge']] * 0.3
    age_score = 3 if answers['age'] < 45 else 0
    total_score = horizon_score + loss_score + knowledge_score + age_score
    return "Conservative" if total_score < 5.0 else "Moderate" if 5.0 <= total_score < 8.0 else "Aggressive"

def optimize_portfolio(risk_profile, data):
    if data.empty or len(data.columns) < 3: return {}
    try:
        mu, S = expected_returns.capm_return(data), risk_models.CovarianceShrinkage(data).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        if risk_profile == "Conservative":
            ef.add_constraint(lambda w: w <= 0.15)
            ef.min_volatility()
        elif risk_profile == "Moderate":
            selected = (data.pct_change(90).mean() / data.pct_change().std()).nlargest(8).index.tolist()
            ef = EfficientFrontier(mu[selected], S.loc[selected, selected])
            ef.add_objective(objective_functions.L2_reg)
            ef.max_sharpe()
        else:
            crypto_assets = [t for t in data.columns if ASSET_UNIVERSE.get(t) == 'Crypto']
            if crypto_assets: 
                ef.add_constraint(lambda w: sum(w[c] for c in crypto_assets) >= 0.35)
            ef.add_objective(objective_functions.L2_reg, gamma=0.005)
            ef.max_sharpe()
        return ef.clean_weights()
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

def calculate_metrics(weights, data):
    try:
        returns = data.pct_change().dropna()
        valid_assets = [a for a in weights if a in returns.columns]
        aligned_weights = np.array([weights[a] for a in valid_assets])
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

def generate_response(prompt, data, metrics, weights):
    try:
        context = f"""
        Portfolio Metrics:
        - Annual Return: {metrics.get('annual_return', 0):.1%}
        - Volatility: {metrics.get('annual_volatility', 0):.1%}
        - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        - 95% VaR: {metrics.get('var_95', 0):.1f}%
        - 95% CVaR: {metrics.get('cvar_95', 0):.1f}%
        - Max Drawdown: {metrics.get('max_drawdown', 0):.1%}
        
        Portfolio Allocation: {weights}
        
        User Question: {prompt}
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a portfolio risk analyst. Use these metrics to answer:" + context},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Sidebar configuration
with st.sidebar:
    st.header("Investor Profile")
    risk_answers = {
        'horizon': st.slider("Investment Horizon (Years)", 1, 10, 5),
        'loss_tolerance': st.select_slider("Max Loss Tolerance", 
                                         options=["0-10%", "10-20%", "20-30%", "30%+"],
                                         value="20-30%"),
        'knowledge': st.radio("Market Experience", ["Novice", "Intermediate", "Expert"]),
        'age': st.number_input("Age", 18, 100, 40)
    }
    risk_profile = calculate_risk_profile(risk_answers)
    st.metric("Your Risk Profile", risk_profile)
    investment = st.number_input("Investment Amount (€)", 1000, 1000000, 200000)

# Main app logic
if st.button("Generate Portfolio"):
    with st.spinner("Constructing optimal allocation..."):
        data = fetch_data()
        if data.empty: 
            st.stop()
        weights = optimize_portfolio(risk_profile, data)
        if not weights: 
            st.stop()
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

        # Monte Carlo Projections
        if investment > 0 and metrics.get('annual_return'):
            with st.expander(f"Monte Carlo Projections - €{investment:,.0f}", expanded=False):
                fig, ax = plt.subplots(figsize=(12, 7))
                for years, color, alpha in zip([3, 5, 7, 10], ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'], [0.15, 0.12, 0.09, 0.06]):
                    simulations = 500
                    daily_returns = np.random.normal(
                        metrics['annual_return']/252,
                        metrics['annual_volatility']/np.sqrt(252),
                        (252*years, simulations)
                    )
                    growth = investment * np.exp(np.cumsum(daily_returns, axis=0))
                    growth_df = pd.DataFrame(growth)
                    upper, lower = growth_df.quantile(0.9, axis=1), growth_df.quantile(0.1, axis=1)
                    median_growth = growth_df.median(axis=1)
                    
                    ax.fill_between(range(len(median_growth)), lower, upper, color=color, alpha=alpha, label=f'{years}Y 80% Range')
                    ax.plot(median_growth, color=color, linewidth=2.8, alpha=0.95, label=f'{years}Y Median')
                    
                    vertical_offset = 40 if years in [3,7] else -40
                    ax.annotate(
                        f"€{median_growth.iloc[-1]/1e6:.2f}M" if median_growth.iloc[-1] >= 1e6 else f"€{median_growth.iloc[-1]/1e3:.0f}K",
                        xy=(len(median_growth)-1, median_growth.iloc[-1]),
                        xytext=(35, vertical_offset),
                        textcoords='offset points',
                        color=color,
                        fontsize=10,
                        weight='bold',
                        ha='left',
                        va='center',
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=1,
                            alpha=0.6
                        ),
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            fc='white',
                            ec=color,
                            lw=1,
                            alpha=0.9
                        )
                    )

                ax.set_ylim(bottom=0, top=investment*10)
                ax.set_xlim(0, 252*10 + 100)
                ax.set_title("Monte Carlo Projections with Confidence Bounds", fontsize=14, pad=25)
                ax.set_xlabel("Trading Days", fontsize=12)
                ax.set_ylabel("Portfolio Value (€)", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.2)
                ax.legend(loc='upper left', frameon=True, facecolor='white')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'€{x/1e6:.1f}M' if x >= 1e6 else f'€{x/1e3:.0f}K'))
                st.pyplot(fig)

# Conversational interface
user_query = st.text_input("Ask me anything about your portfolio:")
if user_query:
    if 'data' in locals() and 'metrics' in locals() and 'valid_weights' in locals():
        response = generate_response(user_query, data, metrics, valid_weights)
        st.write("**Response:**")
        st.write(response)
        
        if "graph" in user_query.lower() or "plot" in user_query.lower():
            fig, ax = plt.subplots()
            for asset in data.columns:
                ax.plot(data[asset].pct_change().cumsum(), label=asset)
            ax.set_title("Historical Cumulative Returns")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Returns")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.2)
            st.pyplot(fig)
    else:
        st.warning("Please generate a portfolio first before asking questions!")
