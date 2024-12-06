import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("depresion.csv")
df = pd.DataFrame(data)

# Prepare the features and target
X = df[['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours']]
y = df['Depression']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and calculate accuracy
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

# Streamlit Web Application
st.set_page_config(page_title="Logistic Regression Model", page_icon=":guardsman:", layout="wide")

# Title
st.title("Depression Prediction Model - Logistic Regression")

# Explanation text
st.write(
    """
    This app uses a Logistic Regression model to predict depression levels based on features like age, academic pressure, CGPA, study satisfaction, and study/work hours.
    The model has been trained and tested on a dataset, and the accuracy is displayed below.
    """
)

# Displaying the model accuracy
st.subheader("Model Accuracy:")
st.markdown(f"**Accuracy:** {accuracy * 100:.2f}%")

