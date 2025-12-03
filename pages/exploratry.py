import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Exploratory Data Analysis")

uploaded = st.file_uploader("Upload Walmart CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Plot 1
    if "Weekly_Sales" in df:
        st.subheader("Weekly Sales Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Weekly_Sales"], bins=50)
        st.pyplot(fig)

    # Plot 2
    if "Date" in df and "Weekly_Sales" in df:
        st.subheader("Sales Over Time")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Weekly_Sales"])
        ax.set_xticks([])
        st.pyplot(fig)

else:
    st.info("Please upload dataset to view EDA.")
