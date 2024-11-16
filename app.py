import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  # or use pickle to load the saved model

# Title of the app
st.write("""
# Liver Disease Prediction App

This app predicts the likelihood of **Liver Disease** using a pre-trained model!
""")
st.write('---')

# Load the pre-trained model (assumed saved as 'liver_disease_model.pkl')
model = joblib.load('model_dt.pkl')  # Ensure this file is in the same directory or provide the full path

# Load the pre-fitted scaler (assumed saved as 'scaler.pkl')
scaler = joblib.load('scaler_dt.pkl')  # The scaler used during model training

# Sidebar for user input
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 10, 100, 50)  # Example ranges, modify as needed
    direct_bilirubin = st.sidebar.slider('Direct Bilirubin', 0.0, 5.0, 1.0)
    alkaline_phosphotase = st.sidebar.slider('Alkaline Phosphotase', 100, 500, 200)
    aspartate_aminotransferase = st.sidebar.slider('Aspartate Aminotransferase', 10, 500, 50)
    albumin_and_globulin_ratio = st.sidebar.slider('Albumin and Globulin Ratio', 0.0, 3.0, 1.0)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    
    # Convert gender to numerical value (1 for Male, 0 for Female)
    gender_value = 1 if gender == 'Male' else 0
    
    # Create a DataFrame for the features
    data = {'Age': age,
            'Direct_Bilirubin': direct_bilirubin,
            'Alkaline_Phosphotase': alkaline_phosphotase,
            'Aspartate_Aminotransferase': aspartate_aminotransferase,
            'Albumin_and_Globulin_Ratio': albumin_and_globulin_ratio,
            'Male': gender_value}
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input data
df = user_input_features()

# Display the input features
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Scale the user input features
df_scaled = scaler.transform(df)

# Use the loaded model to make predictions
prediction = model.predict(df_scaled)
prediction_proba = model.predict_proba(df_scaled)

# Display prediction results
st.header('Prediction of Liver Disease')
if prediction == 1:
    st.write("The patient is  have liver disease.")
else:
    st.write("The patient is dont have liver disease.")

# Display prediction probabilities
st.subheader('Prediction Probability')
st.write(prediction_proba)
