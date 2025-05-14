import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor and standard scaler pickle file
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        data = [[float(x) for x in request.form.values()]]
        final_input = standard_scaler.transform(np.array(data))
        prediction = ridge_model.predict(final_input)
        return render_template("predict.html", prediction_text=f"Predicted value: {prediction[0]}")
    else:
        return render_template("predict.html")
        
if __name__ == "__main__":
    app.run(host="0.0.0.0")