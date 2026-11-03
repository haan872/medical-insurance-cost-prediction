import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🩺")
st.title("Medical Insurance Cost Prediction")
st.write("Enter the details below to estimate medical insurance charges.")

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_URL)
    encoded = pd.get_dummies(
        df,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int
    )
    X = encoded.drop("charges", axis=1)
    y = encoded["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions),
    }
    return model, X.columns, metrics

try:
    model, feature_columns, metrics = train_model()
except Exception as e:
    st.error("The dataset could not be loaded. Check your internet connection and try again.")
    st.exception(e)
    st.stop()

with st.form("prediction_form"):
    st.subheader("Enter user information")
    age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
    sex = st.selectbox("Gender", ["female", "male"])
    smoker = st.selectbox("Smoking status", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    submitted = st.form_submit_button("Predict Insurance Cost")

if submitted:
    new_person = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])
    new_encoded = pd.get_dummies(
        new_person,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int
    )
    new_encoded = new_encoded.reindex(columns=feature_columns, fill_value=0)
    estimated_cost = float(model.predict(new_encoded)[0])
    st.success(f"Estimated insurance cost: ${estimated_cost:,.2f}")
    st.caption("This is a model estimate for educational purposes, not an official insurance quote.")

with st.expander("Model evaluation information"):
    st.write("The app trains a Linear Regression model using the Medical Cost Personal Dataset.")
    st.write("Evaluation metrics calculated on a held-out test set:")
    st.json({k: round(float(v), 4) for k, v in metrics.items()})
