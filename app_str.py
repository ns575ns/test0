import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Set title of the app
st.title("Admissions Prediction App")

# Add input fields for the user
gre = st.number_input('Enter your GRE score', min_value=0, max_value=800, step=1)
gpa = st.number_input('Enter your GPA', min_value=0.0, max_value=4.0, step=0.01)
rank = st.selectbox('Select your university rank (1 is the best)', options=[1, 2, 3, 4])

# When the "Predict" button is clicked, the model will make a prediction
if st.button("Predict"):
    # Create a dataframe for the input values
    input_data = pd.DataFrame([[gre, gpa, rank]], columns=['gre', 'gpa', 'rank'])
    
    # Preprocess input data to match model training (One-Hot Encode 'rank')
    input_data['rank'] = input_data['rank'].astype('category')
    input_data = pd.get_dummies(input_data, columns=['rank'], drop_first=True)
    
    # Ensure that missing rank columns (e.g., rank_2, rank_3, rank_4) are filled with 0
    for i in range(2, 5):  # rank_2, rank_3, rank_4
        if f'rank_{i}' not in input_data.columns:
            input_data[f'rank_{i}'] = 0
    
    # Make a prediction using the trained model
    prediction = model.predict(input_data)[0]
    
    # Display the prediction result
    if prediction == 1:
        st.success("The student is predicted to be admitted.")
    else:
        st.error("The student is predicted to NOT be admitted.")
