import pandas as pd
import pickle
from flask import Flask, jsonify, request, render_template
import joblib
from xgboost import XGBClassifier
xgb_model = XGBClassifier()


app = Flask(__name__)   #initialize flask app

# load the model from disk
model = pickle.load(open("xgb_model.pkl", 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST': # POST request to handle file upload submission
        # check if the post request has the file part
        # pass dataframe to model
        # one row. model: saved column names put into model.predict
        req = request.form
        #print(req)

        missing = list()
        # loop through dict of req items
        for k, v in req.items():
            if v == "":
                missing.append(k)

        if missing:
            #print("in missing")
            feedback = "Please fill in all values."
            return render_template("index.html", feedback=feedback)
        else:
            gender = int(req['gender'])  #int
            age = round(float(req['age']),2)      #float
            hyper_tension = int(req['hyper_tension'])  #int
            heart_disease = int(req['heart_disease'])   #int
            ever_married = int(req['ever_married'])    #int
            work_type = int(req['work_type'])           #int
            Residence_type = int(req['Residence_type'])  #int
            avg_glucose_level =  round(float(req['avg_glucose_level']),6)  #float round 
            bmi = round(float(req['bmi']),2)          #float round 
            smoking_status = int(req['smoking_status'])    # int

            
            #create dataframe of values
            stroke_dict = {}
            stroke_dict = {
                "gender" : gender,
                "age" : age,
                "hyper_tension" : hyper_tension,
                "heart_disease" : heart_disease,
                "ever_married" : ever_married,
                "work_type" : work_type,
                "Residence_type" : Residence_type,
                "avg_glucose_level" : avg_glucose_level,
                "bmi" : bmi,
                "smoking_status" : smoking_status
            }

        
            stroke_list =[]
            stroke_list.append(stroke_dict)
            #print(stroke_list)

            #put into a datframe to go  into the model
            stroke_df = pd.DataFrame(stroke_list)

            # add the first occurance only
            arr = model.predict(stroke_df)[0]
            #print(arr)
            feedback = "Sorry, no results at this time."
            try:
                if float(arr) >= 0.5:
                    feedback = "\U0001f92d" + "Model predicted: At risk."
                else:
                    feedback = "\U0001f600" + " Model predicted: Not at risk."
            except:
                feedback = "Sorry, no results at this time. Please try again."
           
           
        return render_template("index.html", feedback=feedback)
           
    return render_template('index.html') # GET request yields upload page
    

@app.route('/about')
def about():
    return render_template('about.html')
    

if __name__ == "__main__":
    app.run()
