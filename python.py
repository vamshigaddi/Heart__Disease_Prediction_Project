import streamlit as st
import pickle
import numpy as np

# Load the trained machine learning model
model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))
# Create a Streamlit app
st.title("House Price Prediction App")

# Create a Streamlit app
st.title("House Price Prediction App")

# Create user interface
st.header("Enter House Details:")
sqft = st.slider("Square Feet", 500, 5000, 1500)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
location = st.text_input("Location", "Downtown")

# Make Predictions
if st.button("Predict Price"):
    input_data = np.array([sqft, bedrooms, bathrooms]).reshape(1, -1)
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ${predicted_price:.2f}")

# Optionally, add some additional information or visualizations to your app
st.write("Additional Information:")
st.write("This is a simple house price prediction app.")
