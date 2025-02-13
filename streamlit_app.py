import streamlit as st

st.title('Portfolio Mangement App')

st.info('This app shows your portfolio!')

with st.sidebar:
  st.header('Investment Risk Tolerance Assessment')
  'How would you react if your portfolio lost 20% in a downturn?' = st. selectbox('Sell immediately','Hold and wait',' Buy more because prices are cheaper')
  
  
