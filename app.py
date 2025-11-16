from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("house_price_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "TensorFlow House Price Prediction API on Railway is LIVE!"

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
    app.run(host="0.0.0.0", port=8000)
