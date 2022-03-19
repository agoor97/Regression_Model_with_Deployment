## Import the Libraries
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request
import joblib, os
from utils import process_one, process_batch   ## the function I craeted to process the data in utils.py



## Intialize the Flask APP
app = Flask(__name__)
app.config['UPLOAD_FOLDER1'] = 'static'

## Loading the Model
model = joblib.load('xgboost_model.pkl')


## Route for Home page
@app.route('/')
def home():
    return render_template('index.html')


## Route for predict only one Instances
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        iri0 = float(request.form['iri0'])
        age = float(request.form['age'])
        fc = float(request.form['fc'])
        lc = float(request.form['lc'])
        tc = float(request.form['tc'])
        rut = float(request.form['rut'])

        X_new = [iri0, age, fc, lc, tc, rut]
        X_new = process_one(X_new)
        outs = np.exp(model.predict(X_new))  ## predicting
        outs = '{:.4f}'.format(outs[0])

        return render_template('predict.html', pred_value=outs)  ## outs: predicted value
    else:
        return render_template('predict.html')



## Route for predicting batch of instances exsists in xlsx file.
@app.route('/predict_batch', methods=['GET', 'POST'])
def predict_batch():
    return render_template('index.html')


## Route for about_us page
@app.route('/about_us')
def about_us():


    return render_template('about.html')

## Run the App
if __name__ == '__main__':
    app.run(debug=True)
