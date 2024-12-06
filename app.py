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
st.write(
age = st.number_input("Age", min_value=10, max_value=100, step=1, value=20)
academic_pressure = st.slider("Academic Pressure (1-10)", min_value=1, max_value=10, value=5)
cgpa = st.slider("CGPA (0-10)", min_value=0.0, max_value=10.0, step=0.1, value=7.5)
study_satisfaction = st.slider("Study Satisfaction (1-10)", min_value=1, max_value=10, value=5)
work_study_hours = st.number_input("Work/Study Hours per Week", min_value=0, max_value=100, step=1, value=20)
)
if st.button("Predict Depression Level"):
    input_features = np.array([[age, academic_pressure, cgpa, study_satisfaction, work_study_hours]])
    prediction = model.predict(input_features)
    depression_status = "Depressed" if prediction[0] == 1 else "Not Depressed"
    
    # Display the result
    st.write(f"### Prediction: {depression_status}")


# Displaying the model accuracy
st.subheader("Model Accuracy:")
st.markdown(f"**Accuracy:** {accuracy * 100:.2f}%")

