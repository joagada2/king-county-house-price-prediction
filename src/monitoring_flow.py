from datetime import datetime, timedelta
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
import pandas as pd
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter
import datetime
import joblib
import os

#task to load dataframes 1 to 4
@task
def load_first_df():
    new_df_1 = pd.read_csv('../data/monitoring_data/raw_monitoring/new_df_1.csv')
    return new_df_1

@task
def load_second_df():
    new_df_2 = pd.read_csv('../data/monitoring_data/raw_monitoring/new_df_2.csv')
    return new_df_2

@task
def load_third_df():
    new_df_3 = pd.read_csv('../data/monitoring_data/raw_monitoring/new_df_3.csv')
    return new_df_3

@task
def load_fourth_df():
    new_df_4 = pd.read_csv('../data/monitoring_data/raw_monitoring/new_df_4.csv')
    return new_df_4

@task
def get_training_data():
    training_data = pd.read_csv('../data/raw_data/kc_house_data.csv')
    return training_data

#function to process data
@task
def process_data(a):
    a['date'] = pd.to_datetime(a['date'])
    a['month'] = a['date'].apply(lambda date: date.month)
    a['year'] = a['date'].apply(lambda date: date.year)
    a = a.drop('date', axis=1)
    a = a.drop('id', axis=1)
    a = a.drop('zipcode', axis=1)
    return a

@task
def save_processed_df(b):
    b.to_csv('../data/monitoring_data/processed_monitoring_data/d.csv')

@task
def split_data_x(c):
    X_new_df = c.drop('price',axis=1)
    return X_new_df

@task
def split_data_y(d):
    y_new_df = d['price']
    return y_new_df

@task
def save_final_df(e):
    e.to_csv('../data/monitoring_data/final_monitoring/data/g.csv')

@task
def monitor_model(h,h1,h2,h3,h4,i1,i2,i3,i4):
    #split to features and target
    X = h.drop('price', axis=1)
    y = h['price']
    #create list of all the feature df
    df_X = [h1,h2,h3,h4]

    #create list of all target dataframe
    df_y = [i1,i2,i3,i4]

    # create profile
    profile1 = why.log(h1)

    profile_view1 = profile1.view()
    profile_view1.to_pandas()
    print(profile_view1.to_pandas())

    # set authentication & project keys
    os.environ["WHYLABS_DEFAULT_ORG_ID"] = 'enter_org_id'
    os.environ["WHYLABS_API_KEY"] = 'enter_api_key'
    os.environ["WHYLABS_DEFAULT_DATASET_ID"] = 'enter_model_id'

    # Single Profile
    writer = WhyLabsWriter()
    profile = why.log(h1)
    writer.write(file=profile.view())

    # back fill 1 day per batch
    writer = WhyLabsWriter()
    for i, df in enumerate(df_X):
        # walking backwards. Each dataset has to map to a date to show up as a different batch in WhyLabs
        dt = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=i)

        # create profile for each batch of data
        profile = why.log(df).profile()

        # set the dataset timestamp for the profile
        profile.set_dataset_timestamp(dt)
        # write the profile to the WhyLabs platform
        writer.write(file=profile.view())

    #reference profile
    ref_profile = why.log(X).profile()
    writer = WhyLabsWriter().option(reference_profile_name="training_data_profile")
    writer.write(file=ref_profile.view())

    #Logging output
    pred_df_X = df_X
    model = joblib.load('model/loan_default_pred_model.pkl')

    for i, df in enumerate(pred_df_X):
        y_pred = model.predict(df)
        y_prob = model.predict_proba(df)
        pred_scores = []
        pred_classes = []

        for pred in y_pred:
            pred_classes.append(pred)
        df['class_output'] = pred_classes
        for prob in y_prob:
            pred_scores.append(max(prob))
        df['prob_output'] = pred_scores
        print(pred_scores)

    writer = WhyLabsWriter()
    for i, df in enumerate(pred_df_X):
        out_df = df[['class_output', 'prob_output']].copy()
        # walking backwards. Each dataset has to map to a date to show up as a different batch in WhyLabs
        dt = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=i)
        profile = why.log(out_df).profile()

        # set the dataset timestamp for the profile
        profile.set_dataset_timestamp(dt)
        # write the profile to the WhyLabs platform
        writer.write(file=profile.view())

    # Append ground truth data to dataframe
    for i, df in enumerate(pred_df_X):
        df['ground_truth'] = df_y[i]

    # Log performance
    #print(pred_df_X[0])
    for i, df in enumerate(pred_df_X):
        results = why.log_classification_metrics(
            df,
            target_column="ground_truth",
            prediction_column="class_output",
            score_column="prob_output"
        )
        # walking backwards. Each dataset has to map to a date to show up as a different batch in WhyLabs
        dt = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=i)

        profile = results.profile()
        profile.set_dataset_timestamp(dt)

        results.writer("whylabs").write()

schedule = IntervalSchedule(interval=timedelta(hours=24))

with Flow('model_monitoring_pipeline',schedule) as flow:
    new_df_1 = load_first_df()
    new_df_2 = load_second_df()
    new_df_3 = load_third_df()
    new_df_4 = load_fourth_df()
    training_data = get_training_data()

    processed_new_df_1 = process_data(new_df_1)
    processed_new_df_2 = process_data(new_df_2)
    processed_new_df_3 = process_data(new_df_3)
    processed_new_df_4 = process_data(new_df_4)
    training_data = process_data(training_data)

    save_processed_df(processed_new_df_1)
    save_processed_df(processed_new_df_2)
    save_processed_df(processed_new_df_3)
    save_processed_df(processed_new_df_4)

    X_new_df_1 = split_data_x(processed_new_df_1)
    X_new_df_2 = split_data_x(processed_new_df_2)
    X_new_df_3 = split_data_x(processed_new_df_3)
    X_new_df_4 = split_data_x(processed_new_df_4)

    y_new_df_1 = split_data_y(processed_new_df_1)
    y_new_df_2 = split_data_y(processed_new_df_2)
    y_new_df_3 = split_data_y(processed_new_df_3)
    y_new_df_4 = split_data_y(processed_new_df_4)

    save_final_df(X_new_df_1)
    save_final_df(X_new_df_2)
    save_final_df(X_new_df_3)
    save_final_df(X_new_df_4)

    save_final_df(y_new_df_1)
    save_final_df(y_new_df_2)
    save_final_df(y_new_df_3)
    save_final_df(y_new_df_4)

    monitor_model(training_data, X_new_df_1,X_new_df_2,X_new_df_3,X_new_df_4,
                  y_new_df_1,y_new_df_2,y_new_df_3,y_new_df_4)

flow.run()
#connect to prefect 1 cloud
flow.register(project_name='loan-default-prediction')
flow.run_agent()