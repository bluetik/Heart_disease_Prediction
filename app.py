import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache
def load_data():
    dataset = pd.read_csv("heart.csv")  # Replace with the path to your dataset
    return dataset

# Train the model
@st.cache
def train_model(dataset):
    predictors = dataset.drop("target", axis=1)
    target = dataset["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
    
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    accuracy = accuracy_score(model.predict(X_test), Y_test)
    return model, accuracy

# Load data
dataset = load_data()

# Train model
model, accuracy = train_model(dataset)

# Streamlit app
st.title("Heart Disease Prediction")

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 80, 50)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 2.5)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 1)
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])
    
    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(input_df)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.subheader("Prediction")
    st.write("Heart Disease" if prediction[0] == 1 else "No Heart Disease")
    
    st.subheader("Model Accuracy")
    st.write(f"{accuracy * 100:.2f}%")
