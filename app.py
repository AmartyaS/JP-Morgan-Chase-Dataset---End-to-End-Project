# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:08:51 2021

@author: ASUS
"""

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import sklearn

app = Flask(__name__)
model = pickle.load(open('random_forest_grid_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
     if request.method == 'POST':
         Main_Office=request.form['Main_Office']
         if(Main_Office==1):
             Main_Office=1
         else:
             Main_Office=0
         Deposits_2010=float(request.form['2010_Deposits'])
         Deposits_2011=float(request.form['2011_Deposits'])
         Deposits_2012=float(request.form['2012_Deposits'])
         Deposits_2013=float(request.form['2013_Deposits'])
         Deposits_2014=float(request.form['2014_Deposits'])
         Deposits_2015=float(request.form['2015_Deposits'])
         Established_Age=int(request.form['Established_Age'])        
         Acquired_Age=int(request.form['Acquired_Age'])
         prediction=model.predict([[Main_Office,Deposits_2010,Deposits_2011,Deposits_2012,Deposits_2013,Deposits_2014,Deposits_2015,Established_Age,Acquired_Age]])
         output=round(prediction[0],2)
         if output<0:
                    return render_template('index.html',prediction_texts="Sorry, Kindly provide the values in correct manner")
         else:
                return render_template('index.html',prediction_text="Annual Deposit for the year 2016 : {}".format(output))
     else:
            return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)