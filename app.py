from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender= request.form.get('gender'),
            race_ethnicity= request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course= request.form.get('test_preparation_course'),
            reading_score= float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        pred_df= data.get_data_as_data_frame()
        print(pred_df)
        print("before prediction")

        predict_pipeline = PredictPipeline()
        print("mid pipeline")
        
        preds = PredictPipeline().prediction(pred_df)

        # Ensure scalar and JSON/Jinja-friendly
        value = preds.item() if hasattr(preds, 'item') else float(preds[0])
        print("after prediction")
        print(value)


        return render_template('home.html', result=value)

        # result = predict_pipeline.prediction(pred_df)
        # print("after prediction")
        # print(result[0])
        # return render_template('home.html',result=result[0])
    
if __name__ =="__main__":
    app.run(host="0.0.0.0")



