import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def load_data():
    return pd.read_csv("cars.csv")

@st.cache_data
def train_model(df):
    X = df[['Brand', 'Year', 'Transmission', 'Model']]  # Include 'Model' column
    y = df['Price']
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, encoder

def predict(model, encoder, brand, year, transmission, model_name):  # Include model_name parameter
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Year': [year],
        'Transmission': [transmission],
        'Model': [model_name]  # Include model name in input data
    })
    input_encoded = encoder.transform(input_data)
    prediction = model.predict(input_encoded)
    return prediction

def main():
    data = load_data()
    model, encoder = train_model(data)
    st.title("Car Price Prediction")
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Select", ["Make Prediction"])

    if menu == "Make Prediction":
        st.subheader("Make Prediction")
        brand_options = data['Brand'].unique()
        selected_brand = st.selectbox("Select a car brand:", brand_options)
        year_options = data['Year'].unique()
        selected_year = st.selectbox("Select a car year:", year_options)
        transmission_options = data['Transmission'].unique()
        selected_transmission = st.selectbox("Select transmission type:", transmission_options)
        model_options = data['Model'].unique()  # Add model options
        selected_model = st.selectbox("Select a car model:", model_options)

        if st.button("Predict"):
            prediction = predict(model, encoder, selected_brand, selected_year, selected_transmission, selected_model)
            st.write(f"Predicted price for {selected_brand} {selected_model} {selected_year} model with {selected_transmission} transmission: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()