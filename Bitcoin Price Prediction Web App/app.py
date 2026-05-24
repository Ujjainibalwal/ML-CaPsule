import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Cache the model so it doesn't retrain on every interaction
@st.cache_resource
def load_and_train_model():
    bitcoin = pd.read_csv('coin_Bitcoin.csv')
    bitcoin.drop(["Name", "SNo", "Symbol", "Date"], axis=1, inplace=True)

    X = bitcoin.drop(["Marketcap"], axis=1)
    Y = bitcoin["Marketcap"]

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = Lasso()
    model.fit(xtrain, ytrain)
    return model, X.columns.tolist()

Ls, feature_names = load_and_train_model()

r = st.sidebar.radio("Navigation Menu", ["Home", "Bitcoin Price"])

if r == "Home":
    st.write("""
    # Bitcoin Price Prediction
    """)
    st.image("price.png")
    st.subheader("Bitcoin Price")
    st.write(
        "Bitcoin is a widely used cryptocurrency for the digital market. "
        "It is decentralised, meaning it is not owned by any government or company. "
        "Transactions are simple and easy as it doesn't belong to any country. "
        "Records are stored in Blockchain. Bitcoin price is variable and it is "
        "widely used, so it is important to predict the price of it for making any investment."
    )

# Bitcoin Price Prediction
if r == 'Bitcoin Price':
    st.header("Know the Price of Bitcoin")
    High = st.number_input("Highest Price of Bitcoin")
    Low = st.number_input("Lowest Price of Bitcoin")
    Open = st.number_input("Opening Price of Bitcoin")
    Close = st.number_input("Closing Price of Bitcoin")
    volume = st.number_input("Volume of the Bitcoin")

    if st.button("Predict"):
        # Fix: Use a DataFrame with feature names to avoid the sklearn warning
        input_data = pd.DataFrame(
            [[High, Low, Open, Close, volume]],
            columns=feature_names
        )
        ypred = Ls.predict(input_data)
        st.success(f"Your Predicted Bitcoin Marketcap Is ${abs(ypred[0]):,.2f}")
