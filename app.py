import streamlit as st
import pickle
import pandas as pd
import logging
import numpy as np  # For inverse log transformation

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a"  # Append to the log file
)

logging.info("BigMart Sales Prediction App started.")

# Load model and encoders
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    st.error("Error loading the model. Please check the log file.")
    st.stop()

try:
    with open("encoders_scalers.pkl", "rb") as file:
        encoders_scalers = pickle.load(file)
    logging.info("Encoders and scalers loaded successfully.")
except Exception as e:
    logging.error(f"Error loading encoders and scalers: {e}")
    st.error("Error loading encoders. Please check the log file.")
    st.stop()

# Streamlit app UI
st.title("Sales Store Prediction App")



# Home page description
st.markdown("""
# Welcome to the Sales Store Prediction App

This app predicts the sales of products in BigMart stores based on various input features, including product and outlet information.

### How to use this app:
1. Enter the product and outlet details in the respective sections.
2. Click on the **Predict Sales** button.
3. The app will show you the predicted sales in Indian Rupees (₹).

### Features:
- Product details like item identifier, weight, fat content, and price.
- Outlet details like outlet size, location type, and establishment year.

This app uses a machine learning model trained on historical sales data to estimate the sales of a particular product in different BigMart outlets.
""")
st.header("Input Features")
# Create two columns for input
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Product Information")
    item_identifier = st.text_input("Item Identifier (e.g., FDA15, DRC01)")
    item_weight = st.slider("Item Weight (kg)", min_value=0.0, max_value=100.0, step=0.1)
    item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
    item_visibility = st.slider("Item Visibility (%)", min_value=0.0, max_value=100.0, step=0.01)
    item_type = st.selectbox(
        "Item Type", 
        ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods',
         'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks',
         'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood']
    )
    item_mrp = st.number_input("Item MRP (₹)", min_value=0.0, max_value=1000.0, step=0.1)

with col2:
    st.markdown("### Outlet Information")
    outlet_identifier = st.selectbox(
        "Outlet Identifier", 
        ['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045', 
         'OUT017', 'OUT046', 'OUT035', 'OUT019']
    )
    outlet_establishment_year = st.radio(
        "Outlet Establishment Year", 
        [1999, 2009, 1998, 1987, 1985, 2002, 2007, 1997, 2004],
        horizontal=True
    )
    outlet_size = st.radio(
        "Outlet Size", 
        options=["Small", "Medium", "High"], 
        horizontal=True
    )
    outlet_location_type = st.radio(
        "Outlet Location Type", 
        ["Tier 1", "Tier 2", "Tier 3"], 
        horizontal=True
    )
    outlet_type = st.selectbox(
        "Outlet Type", 
        ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"]
    )

st.markdown("---")

# Validation
try:
    item_weight = float(item_weight) if item_weight else 0.0
    item_visibility = float(item_visibility) if item_visibility else 0.0
    item_mrp = float(item_mrp) if item_mrp else 0.0
except ValueError as e:
    logging.warning(f"Invalid input: {e}")
    st.error(f"Invalid input: {e}")
    st.stop()

# Encoding inputs with handling for unknown categories
try:
    encoded_inputs = {
        "Item_Identifier": (
            encoders_scalers['label_encoders']["Item_Identifier"].transform([item_identifier])[0]
            if item_identifier in encoders_scalers['label_encoders']["Item_Identifier"].classes_
            else -1  # Default for unknown identifiers
        ),
        "Item_Weight": item_weight,
        "Item_Fat_Content": (
            encoders_scalers['label_encoders']["Item_Fat_Content"].transform([item_fat_content])[0]
            if item_fat_content in encoders_scalers['label_encoders']["Item_Fat_Content"].classes_
            else -1
        ),
        "Item_Visibility": item_visibility,
        "Item_Type": (
            encoders_scalers['label_encoders']["Item_Type"].transform([item_type])[0]
            if item_type in encoders_scalers['label_encoders']["Item_Type"].classes_
            else -1
        ),
        "Item_MRP": item_mrp,
        "Outlet_Identifier": (
            encoders_scalers['label_encoders']["Outlet_Identifier"].transform([outlet_identifier])[0]
            if outlet_identifier in encoders_scalers['label_encoders']["Outlet_Identifier"].classes_
            else -1
        ),
        "Outlet_Establishment_Year": outlet_establishment_year,
        "Outlet_Size": (
            encoders_scalers['label_encoders']["Outlet_Size"].transform([outlet_size])[0]
            if outlet_size in encoders_scalers['label_encoders']["Outlet_Size"].classes_
            else -1
        ),
        "Outlet_Location_Type": (
            encoders_scalers['label_encoders']["Outlet_Location_Type"].transform([outlet_location_type])[0]
            if outlet_location_type in encoders_scalers['label_encoders']["Outlet_Location_Type"].classes_
            else -1
        ),
        "Outlet_Type": (
            encoders_scalers['label_encoders']["Outlet_Type"].transform([outlet_type])[0]
            if outlet_type in encoders_scalers['label_encoders']["Outlet_Type"].classes_
            else -1
        ),
    }
    logging.info("Inputs encoded successfully.")
except Exception as e:
    logging.error(f"Encoding error: {e}")
    st.error(f"Encoding error: {e}")
    st.stop()

# Prepare input DataFrame
input_df = pd.DataFrame([encoded_inputs])

# Prediction button
if st.button("Predict Sales"):
    try:
        # Prediction from the model (log-transformed sales)
        prediction_log = model.predict(input_df)
        # Reverse log transformation (exponentiate to get original sales value)
        prediction_sales = np.exp(prediction_log[0])  # Apply np.exp to reverse log transformation
        logging.info(f"Prediction successful: {prediction_sales}")
        st.success(f"Predicted Sales: ₹{prediction_sales:,.2f}")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error(f"An error occurred during prediction: {e}")
