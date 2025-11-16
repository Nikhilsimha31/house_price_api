from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load the model
model = tf.keras.models.load_model("house_price_model.h5")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "House Price Prediction API is LIVE on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data, index=[0])

        scaled = scaler.transform(df)
        prediction = model.predict(scaled)
        price = float(prediction[0][0])

        return jsonify({"predicted_price": price})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()
