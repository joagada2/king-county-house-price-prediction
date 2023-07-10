import json

import requests
import streamlit as st

def get_inputs():
    """Get inputs from users on streamlit"""
    st.title("Predict House Price")
    st.write("Note: Yes = 1, No= 0")

    data = {}

    data['bedrooms'] = st.number_input(
        'Number of Bedrooms'
    )
    data['bathrooms'] = st.number_input(
        'Number of Bathrooms'
    )
    data['sqft_living'] = st.number_input(
        'Size of Living Room in Sqft'
    )
    data['sqft_lot '] = st.number_input(
        'Size of Parking Lot in Sqft'
    )
    data['floors'] = st.number_input(
        'Number of Floors'
    )
    data['waterfront'] = st.selectbox(
        'Is House in Waterfront?',
        options=[0, 1],
    )
    data['view'] = st.number_input(
    'view'
    )
    data['condition'] = st.number_input(
    'condition'
    )
    data['grade'] = st.number_input(
    'grade'
    )
    data['sqft_above'] = st.number_input(
    'sqft_above'
    )
    data['sqft_basement'] = st.number_input(
        'Size of Basement in Sqft'
    )
    data['yr_built'] = st.number_input(
        'Year Built'
    )
    data['yr_renovated'] = st.number_input(
        'Year Renovated'
    )
    data['lat'] = st.number_input(
    'Latitude'
    )
    data['long'] = st.number_input(
        'Longitude'
    )
    data['sqft_living15'] = st.number_input(
        'sqft_living15'
    )
    data['sqft_lot15'] = st.number_input(
        'sqft_lot15'
    )
    data['month'] = st.number_input(
        'month'
    )
    data['year'] = st.number_input(
        'year'
    )
    return data

def write_predictions(data: dict):
    if st.button("House Price?"):
        data_json = json.dumps(data)

        prediction = requests.post(
            "https://House-price-prediction-app.herokuapp.com/predict",
            headers={"content-type": "application/json"},
            data=data_json,
        ).text[0]

        st.write({prediction})

def main():
    data = get_inputs()
    write_predictions(data)

if __name__ == "__main__":
    main()


# streamlit run app_name.py