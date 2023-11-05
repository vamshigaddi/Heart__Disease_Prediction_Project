import streamlit as st
import pickle
import numpy as np
import json

# Load the trained machine learning model
model = pickle.load(open('bangalore_home_prices_model.pickle', 'rb'))

# Load the JSON file with column names
with open("columns.json", "r") as f:
    columns = json.load(f)

data_columns = columns.get('data_columns', [])

# Set a colorful background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #f6e2e2, #b7c0cd);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a Streamlit app
st.title("House Price Prediction App")

# Custom CSS to style the header
st.markdown(
    """
    <style>
    .stHeader {
        color: #e36d2d;
        font-size: 36px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS to style the main content
st.markdown(
    """
    <style>
    .stMarkdown {
        color: #333;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create user interface
st.markdown("## Enter House Details:")
sqft = st.selectbox("Square Feet", range(500, 5001, 100), index=30)
bedrooms = st.selectbox("Bedrooms", range(1, 11), index=2)
bathrooms = st.selectbox("Bathrooms", range(1, 11), index=2)

# Dropdown for Location
location = st.selectbox("Select Location", data_columns)

# Search option for Location
custom_location = st.text_input("Or Enter Custom Location")

# Combine selected location and custom location
selected_location = location if location else custom_location

# Make Predictions
if st.button("Predict Price"):
    # Create input data
    input_data = [sqft, bedrooms, bathrooms, selected_location]
    input_data = np.array(input_data).reshape(1, -1)

    # Perform any necessary preprocessing on the input data before making predictions
    # For example, you may need to one-hot encode the location feature

    # Make predictions
    predicted_price = model.predict(input_data)[0]
    st.markdown(f"## Estimated House Price: ${predicted_price:.2f}")

# Optionally, add some additional information or visualizations to your app
st.markdown("### Additional Information:")
st.markdown("This is a colorful house price prediction app that will surprise you!")
