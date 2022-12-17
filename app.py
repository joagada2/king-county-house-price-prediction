from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
import pandas as pd
from wtforms import StringField,SubmitField
from wtforms.validators import NumberRange

import numpy as np  
from tensorflow.keras.models import load_model
import joblib



def predict_house_price(model,scaler,sample_json):
    
    a = sample_json['bedrooms']
    b = sample_json['bathrooms']
    c = sample_json['sqft_living']
    d = sample_json['sqft_lot']
    e = sample_json['floors']
    f = sample_json['waterfront']
    g = sample_json['view']
    h = sample_json['condition']
    i = sample_json['grade']
    j = sample_json['sqft_above']
    k = sample_json['sqft_basement']
    l = sample_json['yr_built']
    m = sample_json['yr_renovated']
    n = sample_json['lat']
    o = sample_json['long']
    p = sample_json['sqft_living15']
    q = sample_json['sqft_lot15']
    r = sample_json['month']
    s = sample_json['year']
    
    columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade',
              'sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15','sqft_lot15','month',
              'year']
    
    house = [[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s]]
    
    house = pd.DataFrame(house, columns=columns)
    
    house = scaler.transform(house.values.reshape(-1, 19))
    
    prediction = model.predict(house)
    
    return float(prediction)

app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'

# REMEMBER TO LOAD THE MODEL AND THE SCALER!
model = load_model("county_house_price_prediction_model.h5")
scaler = joblib.load("county_house_price_prediction_scaler.pkl")

# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class FlowerForm(FlaskForm):
    a1 = StringField('Number of bedroom')
    b1 = StringField('Number of bathrooms')
    c1 = StringField('Size of living room in sqft')
    d1 = StringField('Size of parking lot in sqft')
    e1 = StringField('Floor')
    f1 = StringField('Waterfront')
    g1 = StringField('Views')
    h1 = StringField('Condition')
    i1 = StringField('Grade')
    j1 = StringField('Size of space above ground level(in sqft)')
    k1 = StringField('Size of basement()in sqft')
    l1 = StringField('Year house was built')
    m1 = StringField('Year house was renovated last')
    n1 = StringField('Latitude')
    o1 = StringField('Longitude')
    p1 = StringField('Size of interior for 15 nearest neighbour(sqft)')
    q1 = StringField('Size of parking lot for 15 nearest neighbour (sqft)')
    r1 = StringField('Month the house is to be sold')
    s1 = StringField('Year the house is to be sold')

    submit = SubmitField('Predict')

@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['a1'] = form.a1.data
        session['b1'] = form.b1.data
        session['c1'] = form.c1.data
        session['d1'] = form.d1.data
        session['e1'] = form.e1.data
        session['f1'] = form.f1.data
        session['g1'] = form.g1.data
        session['h1'] = form.h1.data
        session['i1'] = form.i1.data
        session['j1'] = form.j1.data
        session['k1'] = form.k1.data
        session['l1'] = form.l1.data
        session['m1'] = form.m1.data
        session['n1'] = form.n1.data
        session['o1'] = form.o1.data
        session['p1'] = form.p1.data
        session['q1'] = form.q1.data
        session['r1'] = form.r1.data
        session['s1'] = form.s1.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    
    content = {}

    content['bedrooms'] = float(session['a1'])
    content['bathrooms'] = float(session['b1'])
    content['sqft_living'] = float(session['c1'])
    content['sqft_lot'] = float(session['d1'])
    content['floors'] = float(session['e1'])
    content['waterfront'] = float(session['f1'])
    content['view'] = float(session['g1'])
    content['condition'] = float(session['h1'])
    content['grade'] = float(session['i1'])
    content['sqft_above'] = float(session['j1'])
    content['sqft_basement'] = float(session['k1'])
    content['yr_built'] = float(session['l1'])
    content['yr_renovated'] = float(session['m1'])
    content['lat'] = float(session['n1'])
    content['long'] = float(session['o1'])
    content['sqft_living15'] = float(session['p1'])
    content['sqft_lot15'] = float(session['q1'])
    content['month'] = float(session['r1'])
    content['year'] = float(session['s1'])

    results = predict_house_price(model=model,scaler=scaler,sample_json=content)

    return render_template('prediction.html',results=results)

if __name__ == '__main__':
    app.run(debug=True)