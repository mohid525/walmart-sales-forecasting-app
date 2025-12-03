import streamlit as st

st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

st.sidebar.success("Select a page above")

st.title("Walmart Sales Forecasting Dashboard")

st.write("""
Welcome to the Walmart Sales Forecasting Application.

Use the sidebar to navigate:
- Introduction  
- Exploratory Data Analysis  
- Predictions  
- Conclusion  
""")
