import streamlit as st

st.title('Portfolio Mangement App')

st.info('This app shows your portfolio!')

with st.sidebar:
  st.header('Investment Risk Tolerance Assessment')
  Downturn = st. selectbox('Downturn',('','Sell immediately','Hold and wait',' Buy more because prices are cheaper'))
  Goal = st. selectbox('Goal',('','Saftey','Growth','High Returns'))
  Experience = st. selectbox ('Experience', ('','Beginner','Intermediate','Expert'))
  
def calculate_risk_score(Downturn, Goal, Experience):
    score = 0

    # Downturn Reaction
    downturn_scores = {
        'Sell immediately': 0,
        'Hold and wait': 2,
        'Buy more because prices are cheaper': 4
    }
    score += downturn_scores.get(Downturn, 0)

    # Investment Goal
    goal_scores = {
        'Safety': 0,
        'Growth': 2,
        'High Returns': 4
    }
    score += goal_scores.get(Goal, 0)

    # Investment Experience
    experience_scores = {
        'Beginner': 0,
        'Intermediate': 2,
        'Expert': 4
    }
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
                                   
  
  
