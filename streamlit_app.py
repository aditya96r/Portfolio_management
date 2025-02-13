import streamlit as st
import matplotlib as plt

st.title('Portfolio Mangement App')

st.info('This app shows your portfolio!')

with st.sidebar:
  st.header('Investment Risk Tolerance Assessment')
  Downturn = st. selectbox('Downturn',('','Sell immediately','Hold and wait',' Buy more because prices are cheaper'))
  Goal = st. selectbox('Goal',('','Saftey','Growth','High Returns'))
  Experience = st. selectbox ('Experience', ('','Beginner','Intermediate','Expert'))
  
def calculate_risk_score(Downturn, Goal, Experience):
    score = 0

    downturn_scores = {'Sell immediately': 0, 'Hold and wait': 2, 'Buy more because prices are cheaper': 4}
    goal_scores = {'Safety': 0, 'Growth': 2, 'High Returns': 4}
    experience_scores = {'Beginner': 0, 'Intermediate': 2, 'Expert': 4}

    score += downturn_scores.get(Downturn, 0)
    score += goal_scores.get(Goal, 0)
    score += experience_scores.get(Experience, 0)

    return score
  
# Calculate Risk Score
if Downturn and Goal and Experience:
    risk_score = calculate_risk_score(Downturn, Goal, Experience)

    # Categorize Risk Profile
    if risk_score <= 2:
        risk_category = "Conservative"
    elif risk_score <= 6:
        risk_category = "Moderate"
    else:
        risk_category = "Aggressive"

    # Display result
    st.write(f"### Your Risk Profile: **{risk_category}**")
  # Define Asset Allocation Based on Risk Profile
    if risk_category == "Conservative":
        allocation = {"Stocks": 20, "Bonds": 50, "Gold": 20, "Cash": 10}
    elif risk_category == "Moderate":
        allocation = {"Stocks": 50, "Bonds": 30, "Gold": 15, "Cash": 5}
    else:  # Aggressive
        allocation = {"Stocks": 80, "Bonds": 10, "Gold": 5, "Cash": 5}

    # Display Portfolio Allocation in an Expander
    with st.expander("View Portfolio Allocation"):
        st.write(f"**{risk_category} Portfolio Allocation:**")
        for asset, percent in allocation.items():
            st.write(f"- {asset}: **{percent}%**")

        # Ensure allocation is defined before generating pie chart
        if allocation:
            fig, ax = plt.subplots(figsize=(6,6))  
            ax.pie(allocation.values(), labels=allocation.keys(), autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Ensures pie chart is circular
            st.pyplot(fig)
                                   
  
  
