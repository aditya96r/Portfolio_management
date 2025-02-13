import streamlit as st

st.title('Portfolio Mangement App')

st.info('This app shows your portfolio!')

with st.sidebar:
  st.header('Investment Risk Tolerance Assessment')
  'downturn?' = st. selectbox('Downturn',('Sell immediately','Hold and wait',' Buy more because prices are cheaper'))
  
  
