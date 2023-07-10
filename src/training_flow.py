from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from datetime import datetime, timedelta
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
import warnings
warnings.filterwarnings(action="ignore")
import numpy as np
import xgboost as xgb
import joblib
import pandas as pd
from xgboost import XGBRegressor
import os
import mlflow
from dagshub import DAGsHubLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, \
    mean_absolute_percentage_error

@task
def get_data():
    data = pd.read_csv('data/raw_data/kc_house_data.csv')
    return data

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
def train_model(b):
    X = b.drop('price', axis=1)
    y = b['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=43)
    X_test.to_csv('data/test_data/X_test.csv',index=False)
    y_test.to_csv('data/test_data/y_test.csv',index=False)
    model = xgb.XGBRegressor(eval_metric='rmsle')
    param_grid = {"max_depth": [5, 6, 7],
                  "n_estimators": [600, 700, 800],
                  "learning_rate": [0.01, 0.015, 0.020]}
    search = GridSearchCV(model, param_grid, cv=5).fit(X_train, y_train)

    print("The best hyperparameters are ", search.best_params_)

    model = xgb.XGBRegressor(learning_rate=search.best_params_["learning_rate"],
                                 n_estimators=search.best_params_["n_estimators"],
                                 max_depth=search.best_params_["max_depth"],
                                 eval_metric='rmsle')

    model.fit(X_train, y_train)
    # Save model to model folder
    joblib.dump(model, 'model/king_county_house_price_prediction_model.pkl')

    # save model to app dir for deployment
    joblib.dump(model, 'app/model1/king_county_house_price_prediction_model.pkl')
    return model

@task
def evaluate(c):
    #link up to dagshub MLFlow environment
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/joe88data/king-county-house-price-prediction.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'joe88data'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'e94114ca328c75772401898d749decb6dbcbeb21'
    with mlflow.start_run():
        # Load data and model
        X_test = pd.read_csv('data/test_data/X_test.csv')
        y_test = pd.read_csv('data/test_data/y_test.csv')
        c = joblib.load('model/king_county_house_price_prediction_model.pkl')

        # Get predictions
        prediction = c.predict(X_test)

        # Get metrics
        RMSE = np.sqrt(MSE(y_test, prediction))
        print("The RMSE score for this model is %f" % RMSE)

        RMSLE = np.sqrt(mean_squared_log_error(y_test, prediction))
        print("The RMSLE score for this model is %f" % RMSLE)

        MAE = mean_absolute_error(y_test, prediction)
        print("The MAE score for this model is %f" % MAE)

        MAPE = mean_absolute_percentage_error(y_test, prediction)
        print("The MAPE score for this model is %f" % MAPE)

        R_Squared = r2_score(y_test, prediction)
        print("The R_Squared score for this model is %f" % R_Squared)

        # helper class for logging model and metrics
        class BaseLogger:
            def __init__(self):
                self.logger = DAGsHubLogger()

            def log_metrics(self, metrics: dict):
                mlflow.log_metrics(metrics)
                self.logger.log_metrics(metrics)

            def log_params(self, params: dict):
                mlflow.log_params(params)
                self.logger.log_hyperparams(params)
        logger = BaseLogger()
        # function to log parameters to dagshub and mlflow
        def log_params(c: XGBRegressor):
            logger.log_params({"model_class": type(c).__name__})
            model_params = c.get_params()

            for arg, value in model_params.items():
                logger.log_params({arg: value})

        # function to log metrics to dagshub and mlflow
        def log_metrics(**metrics: dict):
            logger.log_metrics(metrics)
        # log metrics to remote server (dagshub)
        log_params(c)
        log_metrics(RMSE=RMSE, RSuared=R_Squared, RMSLE=RMSLE, MAE=MAE, MAPE=MAPE)
            # log metrics to local mlflow
            # mlflow.sklearn.log_model(model, "model")
            # mlflow.log_metric('f1_score', f1)
            # mlflow.log_metric('accuracy_score', accuracy)
            # mlflow.log_metric('area_under_roc', area_under_roc)
            # mlflow.log_metric('precision', precision)
            # mlflow.log_metric('recall', recall)

#adding schedule here automate the pipeline and make it run every 10 minutes
schedule = IntervalSchedule(interval=timedelta(minutes=10))

#create and run flow locally. To schedule to workflow to be automatically triggered every 4 hrs,
#add 'schedule' as Flow parameter (ie with Flow("loan-default-prediction", schedule)
with Flow("training_pipeline") as flow:
    data = get_data()
    processed_data = process_data(data)
    model = train_model(processed_data)
    evaluate(model)

#flow.visualize()
flow.run()
#connect to prefect 1 cloud
flow.register(project_name='king-county-house-price-prediction')
flow.run_agent()

