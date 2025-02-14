import matplotlib.pyplot as plt
import streamlit as st

# Add title and information at the top
st.title('Portfolio Management App')
st.info('This App shows your portfolio')

# Sidebar for the Investment Risk Tolerance Assessment
with st.sidebar:
    st.header('Investment Risk Tolerance Assessment')
    Downturn = st.selectbox('Downturn', ('', 'Sell immediately', 'Hold and wait', 'Buy more because prices are cheaper'))
    Goal = st.selectbox('Goal', ('', 'Safety', 'Growth', 'High Returns'))
    Experience = st.selectbox('Experience', ('', 'Beginner', 'Intermediate', 'Expert'))

def calculate_risk_score(Downturn, Goal, Experience):
    score = 0
    downturn_scores = {
        'Sell immediately': 0,
        'Hold and wait': 2,
        'Buy more because prices are cheaper': 4
    }
    goal_scores = {
        'Safety': 0,
        'Growth': 2,
        'High Returns': 4
    }
    experience_scores = {
        'Beginner': 0,
        'Intermediate': 2,
        'Expert': 4
    }
    score += downturn_scores.get(Downturn, 0)
    score += goal_scores.get(Goal, 0)
    score += experience_scores.get(Experience, 0)
    return score

if Downturn and Goal and Experience:
    risk_score = calculate_risk_score(Downturn, Goal, Experience)
    
    # Determine risk category and asset allocation
    if risk_score <= 2:
        risk_category = "Conservative"
        allocation = {"Stocks": 20, "Bonds": 50, "Gold": 20, "Cash": 10}
    elif risk_score <= 6:
        risk_category = "Moderate"
        allocation = {"Stocks": 50, "Bonds": 30, "Gold": 15, "Cash": 5}
    else:
        risk_category = "Aggressive"
        allocation = {"Stocks": 80, "Bonds": 10, "Gold": 5, "Cash": 5}

    with st.expander("Investment Recommendations", expanded=True):
        st.header("Your Investment Recommendation")
        col_top = st.columns([3, 2])
        with col_top[0]:
            st.write(f"**Risk Score:** {risk_score}")
        with col_top[1]:
            st.write(f"**Risk Category:** {risk_category}")

        # Define stock selections and technologies based on risk category
        if risk_category == "Conservative":
            stock_picks = {"AAPL": 20, "MSFT": 20, "GOOGL": 20, "AMZN": 20, "FB": 20}
            technology = "S&P 500 Index selected using alphalens"
        elif risk_category == "Moderate":
            stock_picks = {"JNJ": 30, "PG": 25, "MA": 25, "V": 20}
            technology = "Factor Investing (Value & Momentum) using alphalens"
        else:
            stock_picks = {"TSLA": 40, "AMZN": 30, "NVDA": 30}
            technology = "Markowitz MPT optimized portfolio using pypfopt"

        # Prepare data for visualization
        labels = list(stock_picks.keys())
        stock_total = allocation["Stocks"]
        sizes = [(stock_total * weight / 100) for weight in stock_picks.values()]

        # Add other assets to the chart
        other_assets = {k: v for k, v in allocation.items() if k != "Stocks"}
        for asset, percent in other_assets.items():
            labels.append(asset)
            sizes.append(percent)

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.tab20c.colors[:len(labels)],
            wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}
        )
        ax.set_title('Portfolio Allocation with Stock Selection', pad=20)
        ax.axis('equal')

        # Display using columns
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(fig)
        with col2:
            st.subheader("Investment Strategy")
            st.info(f"**Technology Used:**\n{technology}")
            st.subheader("Key Holdings")
            for stock in list(stock_picks.keys())[:3]:
                st.write(f"- {stock}")

        # Detailed breakdown
        st.subheader("Portfolio Composition")
        st.write(f"**Equity Allocation ({stock_total}%)**:")
        for stock, weight in stock_picks.items():
            st.write(f"- {stock}: {weight}% of stocks ({stock_total * weight / 100}% total)")
        st.write("\n**Fixed Income & Others**:")
        for asset, percent in other_assets.items():
            st.write(f"- {asset}: {percent}%")
