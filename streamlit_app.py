import streamlit as st

st.title('Portfolio Mangement App')

st.info('This app shows your portfolio!')

with st.sidebar:
  st.header('Investment Risk Tolerance Assessment')
  Downturn = st. selectbox('Downturn',('Sell immediately','Hold and wait',' Buy more because prices are cheaper'))
  Goal = st. selectbox('Goal',('Saftey','Growth','High Returns'))
  Experience = st. selectbox ('Experience', ('Beginner','Intermediate','Expert'))
  
                               
  
  
