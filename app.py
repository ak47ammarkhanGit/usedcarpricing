from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            length=float(request.form.get('length')),
            width=float(request.form.get('width')),
            curb_weight=request.form.get('curb_weight'),
            engine_size=request.form.get('engine_size'),
            horsepower=request.form.get('horsepower'),
            city_L_per_100km=float(request.form.get('city_L_per_100km')),
            highway_L_per_100km=float(request.form.get('highway_L_per_100km')),
            wheel_base=float(request.form.get('wheel_base')),
            bore=float(request.form.get('bore')),
            drive_wheels=request.form.get('drive_wheels'),

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)