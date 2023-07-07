import json

import requests
import streamlit as st

def get_inputs():
    """Get inputs from users on streamlit"""
    st.title("Predict Loan Default")
    st.write("Note: Yes = 1, No= 0")

    data = {}

    data['loan_amnt'] = st.number_input(
        'Loan Amount'
    )
    data['term'] = st.number_input(
        'Loan Term'
    )
    data['int_rate'] = st.number_input(
        'Interest Rate'
    )
    data['installment'] = st.number_input(
        'Installment'
    )
    data['annual_inc'] = st.number_input(
        'Annual Income'
    )
    data['dti'] = st.number_input(
        'dti'
    )
    data['pub_rec'] = st.number_input(
    'pub_rec'
    )
    data['revol_bal'] = st.number_input(
    'revol_bal'
    )
    data['revol_util'] = st.number_input(
    'revol_util'
    )
    data['total_acc'] = st.number_input(
    'total_acc'
    )
    data['mort_acc'] = st.number_input(
        'mort_acc'
    )
    data['pub_rec_bankruptcies'] = st.number_input(
        'pub_rec_bankruptcies'
    )
    data['earliest_cr_year'] = st.number_input(
        'earliest_cr_year'
    )
    data['sub_grade_A2'] = st.selectbox(
        'sub_grade_A2?',
         options=[0, 1],
    )
    data['sub_grade_A3'] = st.selectbox(
        'sub_grade_A3?',
        options=[0, 1],
    )
    data['sub_grade_A4'] = st.selectbox(
        'sub_grade_A4?',
        options=[0, 1],
    )
    data['sub_grade_A5'] = st.selectbox(
        'sub_grade_A5?',
        options=[0, 1],
    )
    data['sub_grade_B1'] = st.selectbox(
        'sub_grade_B1?',
        options=[0, 1],
    )
    data['sub_grade_B2'] = st.selectbox(
        'sub_grade_B2?',
        options=[0, 1],
    )
    data['sub_grade_B3'] = st.selectbox(
        'sub_grade_B3?',
        options=[0, 1],
    )
    data['sub_grade_B4'] = st.selectbox(
        'sub_grade_B4?',
        options=[0, 1],
    )
    data['sub_grade_B5'] = st.selectbox(
        'sub_grade_B5?',
        options=[0, 1],
    )
    data['sub_grade_C1'] = st.selectbox(
        'sub_grade_C1?',
        options=[0, 1],
    )
    data['sub_grade_C2'] = st.selectbox(
        'sub_grade_C2?',
        options=[0, 1],
    )
    data['sub_grade_C3'] = st.selectbox(
        'sub_grade_C3?',
        options=[0, 1],
    )
    data['sub_grade_C4'] = st.selectbox(
        'sub_grade_C4?',
        options=[0, 1],
    )
    data['sub_grade_C5'] = st.selectbox(
        'sub_grade_C5?',
        options=[0, 1],
    )
    data['sub_grade_D1'] = st.selectbox(
        'sub_grade_D1?',
        options=[0, 1],
    )
    data['sub_grade_D2'] = st.selectbox(
        'sub_grade_D2?',
        options=[0, 1],
    )
    data['sub_grade_D3'] = st.selectbox(
        'sub_grade_D3?',
        options=[0, 1],
    )
    data['sub_grade_D4'] = st.selectbox(
        'sub_grade_D4?',
        options=[0, 1],
    )
    data['sub_grade_D5'] = st.selectbox(
        'sub_grade_D5?',
        options=[0, 1],
    )
    data['sub_grade_E1'] = st.selectbox(
        'sub_grade_E1?',
        options=[0, 1],
    )
    data['sub_grade_E2'] = st.selectbox(
        'sub_grade_E2?',
        options=[0, 1],
    )
    data['sub_grade_E3'] = st.selectbox(
        'sub_grade_E3?',
        options=[0, 1],
    )
    data['sub_grade_E4'] = st.selectbox(
        'sub_grade_E4?',
        options=[0, 1],
    )
    data['sub_grade_E5'] = st.selectbox(
        'sub_grade_E5?',
        options=[0, 1],
    )
    data['sub_grade_F1'] = st.selectbox(
        'sub_grade_F1?',
        options=[0, 1],
    )
    data['sub_grade_F2'] = st.selectbox(
        'sub_grade_F2?',
        options=[0, 1],
    )
    data['sub_grade_F3'] = st.selectbox(
        'sub_grade_F3?',
        options=[0, 1],
    )
    data['sub_grade_F4'] = st.selectbox(
        'sub_grade_F4?',
        options=[0, 1],
    )
    data['sub_grade_F5'] = st.selectbox(
        'sub_grade_F5?',
        options=[0, 1],
    )
    data['sub_grade_G1'] = st.selectbox(
        'sub_grade_G1?',
        options=[0, 1],
    )
    data['sub_grade_G2'] = st.selectbox(
        'sub_grade_G2?',
        options=[0, 1],
    )
    data['sub_grade_G3'] = st.selectbox(
        'sub_grade_G3?',
        options=[0, 1],
    )
    data['sub_grade_G4?'] = st.selectbox(
        'sub_grade_G4?',
        options=[0, 1],
    )
    data['sub_grade_G5'] = st.selectbox(
        'sub_grade_G5?',
        options=[0, 1],
    )
    data['home_ownership_OWN'] = st.selectbox(
        'home_ownership_OWN?',
        options=[0, 1],
    )

    data['home_ownership_RENT'] = st.selectbox(
        'home_ownership_RENT?',
        options=[0, 1],
    )
    data['home_ownership_OTHER'] = st.selectbox(
        'home_ownership_OTHER?',
        options=[0, 1],

    )
    data['verification_status_Source Verified'] = st.selectbox(
        'verification_status_Source Verified?',
        options=[0, 1],
    )
    data['verification_status_Verified'] = st.selectbox(
        'verification_status_Verified?',
        options=[0, 1],
    )
    data['purpose_credit_card'] = st.selectbox(
        'purpose_credit_card?',
        options=[0, 1],
    )
    data['purpose_debt_consolidation'] = st.selectbox(
        'purpose_debt_consolidation?',
        options=[0, 1],
    )
    data['purpose_educational'] = st.selectbox(
        'purpose_educational?',
        options=[0, 1],
    )

    data['purpose_home_improvement'] = st.selectbox(
        'purpose_home_improvement?',
        options=[0, 1],
    )

    data['purpose_house'] = st.selectbox(
        'purpose_house?',
        options=[0, 1],
    )
    data['purpose_major_purchase'] = st.selectbox(
        'purpose_major_purchase?',
        options=[0, 1],
    )
    data['purpose_medical'] = st.selectbox(
        'purpose_medical?',
        options=[0, 1],
    )
    data['purpose_moving'] = st.selectbox(
        'purpose_moving?',
        options=[0, 1],
    )
    data['purpose_other'] = st.selectbox(
        'purpose_other?',
        options = [0, 1],
    )

    data["purpose_renewable_energy"] = st.selectbox(
        "Male purpose_renewable_energy?",
        options=[0,1],
    )

    data['purpose_small_business'] = st.selectbox(
        'Senior purpose_small_business?',
        options=[0,1],
    )
    data['purpose_vacation'] = st.selectbox(
        'purpose_vacation?',
        options=[0,1],
    )
    data['purpose_wedding'] = st.selectbox(
        'purpose_wedding?',
        options=[0,1],
    )
    data['initial_list_status_w'] = st.selectbox(
        'Phone initial_list_status_w?',
        options=[0, 1],
    )
    data['application_type_INDIVIDUAL'] = st.selectbox(
        'application_type_INDIVIDUAL?',
        options=[0, 1],
    )
    data['application_type_JOINT'] = st.selectbox(
        'application_type_JOINT?',
        options=[0, 1],
    )
    data['zip_code_05113'] = st.selectbox(
        'zip_code = 05113?',
        options=[0, 1],
    )
    data['zip_code_11650'] = st.selectbox(
        'zip_code = 11650?',
        options=[0, 1],
    )
    data['zip_code_22690'] = st.selectbox(
        'zip_code = 22690?',
        options=[0, 1],
    )
    data['zip_code_29597'] = st.selectbox(
        'zip_code = 29597?',
        options=[0, 1],
    )
    data['zip_code_30723'] = st.selectbox(
        'zip_code = 30723?',
        options=[0, 1],
    )
    data['zip_code_48052'] = st.selectbox(
        'zip_code = 48052?',
        options=[0, 1],
    )
    data['zip_code_70466'] = st.selectbox(
        'zip_code = 70466?',
        options=[0, 1],
    )
    data['zip_code_86630'] = st.selectbox(
        'zip_code = 86630',
        options=[0, 1],
    )
    data['zip_code_93700'] = st.selectbox(
        'zip_code = 93700?',
        options=[0, 1],
    )
    return data

def write_predictions(data: dict):
    if st.button("Will this customer default?"):
        data_json = json.dumps(data)

        prediction = requests.post(
            "https://loan-default-prediction-app.herokuapp.com/predict",
            headers={"content-type": "application/json"},
            data=data_json,
        ).text[0]

        if prediction == "0":
            st.write("This customer will default.")
        else:
            st.write("This customer will not default.")

def main():
    data = get_inputs()
    write_predictions(data)

if __name__ == "__main__":
    main()


# streamlit run app_name.py