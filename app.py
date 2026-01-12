# ----------------------------------
# Step 1: Import Required Libraries
# ----------------------------------
import streamlit as st # Streamlit is imported to create web page using python
import numpy as np # numpy is imported to work with numbers and arrays
import pandas as pd # pd is imported to work with tabular data like CSV files
from tensorflow.keras.models import load_model # load_models is used to load the saved models
from sklearn.preprocessing import MinMaxScaler # MixMinScaler is used to scale stock prices (b/w 0-1)

# ----------------------------------
# Step 2: Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Stock Price Prediction App", # Title is shown on the browser tab
    page_icon="📈", # icon is shown in the tab
    layout="centered" # keeps the content in the center of the page
)

st.title("📈 Stock Closing Price Prediction") # Displays the title on the web page
st.write("Predict future stock prices using trained deep learning models")
# Displays the description on the web page


# ----------------------------------
# Step 3: Load Saved Models
# ----------------------------------
@st.cache_resource
def load_models(): # Defines a function to load all trained models.
    model_1d = load_model("models/simple_lstm_best_model_one.keras") # load 1-day prediction model
    model_5d = load_model("models/simple_lstm_best_model_five.keras") # load 5-day prediction model
    model_10d = load_model("models/simple_lstm_best_model_ten.keras") # load 10-day prediction model
    return model_1d, model_5d, model_10d # Returns all three models together.

model_1d, model_5d, model_10d = load_models() # Calls the function and stores each model in a variable.

# ----------------------------------
# Step 4: Load Stock Data
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/TSLA.csv") # This will load the tesla dataset
    df = df[['Adj Close']]  # This will choose the desired column 'Adj Close'
    return df # Returns the cleaned data.

data = load_data() # Calls the function and stores stock prices in data.

# ----------------------------------
# Step 5: Scale Data
# ----------------------------------
scaler = MinMaxScaler() # Creates a scaler object to scale values between 0 and 1.
scaled_data = scaler.fit_transform(data) # Scales the stock prices.

# ----------------------------------
# Step 6: Prepare Latest Input Sequence
# ----------------------------------
WINDOW_SIZE = 60 # The model looks at the last 60 days to make predictions.

latest_sequence = scaled_data[-WINDOW_SIZE:] # latest_sequence = scaled_data[-WINDOW_SIZE:]
latest_sequence = latest_sequence.reshape(1, WINDOW_SIZE, 1) 
# Reshapes data into the format the model expects:
# (samples, time_steps, features)



# ----------------------------------
# Step 7: User Selection
# ----------------------------------
forecast_option = st.selectbox(
    "Select Forecast Horizon",
    ("1-Day", "5-Day", "10-Day") # Creates a dropdown menu.
) # User can choose how many days ahead they want the prediction.

# ----------------------------------
# Step 8: Prediction Logic
# ----------------------------------
if st.button("Predict Closing Price"): # Runs the prediction only when the user clicks the button.

    if forecast_option == "1-Day": # This if block will be executed if user clicks on 1-Day
        prediction = model_1d.predict(latest_sequence) 
        # Uses the 1-day model to predict tomorrow’s price.
        prediction = scaler.inverse_transform(prediction)
        # Converts scaled value back to actual stock price.
        st.success(f"📅 Predicted Closing Price (Tomorrow): ₹ {prediction[0][0]:.2f}")
        # Displays the predicted price nicely on the screen.

    elif forecast_option == "5-Day": # This if block will be executed if user clicks on 5-Day
        prediction = model_5d.predict(latest_sequence) # Predicts prices for the next 5 days.
        prediction = scaler.inverse_transform(prediction) # Converts predictions to real prices.
        st.success("📅 Predicted Closing Prices (Next 5 Days):")
        # Displays the predicted price nicely on the screen.
        for i, price in enumerate(prediction[0], 1): 
            # Loops through each predicted day and displays it.
            st.write(f"Day {i}: ₹ {price:.2f}") 
            # Display the price predictions for 5 days


    elif forecast_option == "10-Day": # This if block will be executed if user clicks on 5-Day
        prediction = model_10d.predict(latest_sequence) # Predicts prices for the next 5 days.
        prediction = scaler.inverse_transform(prediction) # Converts predictions to real prices.
        st.success("📅 Predicted Closing Prices (Next 10 Days):")
        # Displays the predicted price nicely on the screen.
        for i, price in enumerate(prediction[0], 1):
            # Loops through each predicted day and displays it.
            st.write(f"Day {i}: ₹ {price:.2f}") # Display the price predictions for 5 days
