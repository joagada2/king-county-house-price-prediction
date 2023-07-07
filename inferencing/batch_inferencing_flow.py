import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule

#task to load model
@task
def get_data():
    data = pd.read_csv('inferencing/input_data/input_df.csv')
    return data

#task to load model
@task
def load_model():
    model = joblib.load('model/model/king_county_house_price_prediction_model.pkl')
    return model

def process_data(a):
    a['date'] = pd.to_datetime(a['date'])
    a['month'] = a['date'].apply(lambda date: date.month)
    a['year'] = a['date'].apply(lambda date: date.year)
    a = a.drop('date', axis=1)
    a = a.drop('id', axis=1)
    a = a.drop('zipcode', axis=1)
    a.to_csv('inferencing/processed1/processed.csv', index=False)
    return a

#task to get and save predictions
@task
def predict_df(b,c):
    # get prediction
    prediction = c.predict(b)
    b['predicted_prices'] = prediction.tolist()
    #save prediction
    b.to_csv('inferencing/prediction/prediction_df.csv', index=False)

schedule = IntervalSchedule(interval=timedelta(hours=24))

with Flow('batch_inferencing_pipeline') as flow:
    data = get_data()
    model = load_model()
    processed_data = process_data(data)
    predict_df(processed_data, model)

flow.run()
#connect to prefect 1 cloud
flow.register(project_name='king-county-house-price-prediction')
flow.run_agent()
