from flask import Flask,request,app,url_for,render_template
import pickle
import numpy as np
import pandas as pd



### Initiating the app

app = Flask(__name__)

### Loading the  Regressor Model

xgRegressor = pickle.load(open("rfregmodel.pkl","rb"))


### Home Page

@app.route('/')
def home():
    return render_template("index.html")  



### Predict

@app.route('/Submit',methods=['POST'])
def predict():

    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])


    ##data = [float(x) for x in request.form.values()]
    final_input = np.array([[item_weight,item_fat_content,item_visibility,item_mrp,outlet_size,outlet_location_type,outlet_type,item_type,outlet_establishment_year]])
    output = xgRegressor.predict(final_input)[0] 
    ##print(output)
    return render_template("index.html",prediction_text="Your Sales is $ {}".format(output))



if __name__ == "__main__":
 app.run(debug=True)
