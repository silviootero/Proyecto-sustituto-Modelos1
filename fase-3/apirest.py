from flask import Flask, jsonify, request
import numpy as np
from loguru import logger
from joblib import load
import pandas as pd

app = Flask(__name__)

# Estado del entrenamiento
train_status = "not training"

# Cargar el modelo y el escalador
model = load('model.joblib')
scaler = load('scaler.joblib')

# Función de entrenamiento simulada
def _train():
    global train_status
    logger.info("Train started")
    train_status = "training"
    # Simulación del entrenamiento
    sleep(10)
    logger.info("Train finished")
    train_status = "not training"

@app.route("/")
def hello_world():
    return jsonify({"Hello": "World"})

@app.route("/status")
def status():
    return jsonify({"status": train_status})

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Datos recibidos en formato JSON
        data = request.json
        df = pd.DataFrame(data)
        
        # Preprocesamiento de los datos
        df["working_hour"] = ((df["Hour"] >= 5) & (df["Hour"] <= 20)).astype(int)
        df["Seasons"] = df["Seasons"].astype("category").cat.codes
        df["Functioning Day"] = df["Functioning Day"].astype("category").cat.codes
        columns_to_drop = ['Date', 'Snowfall (cm)', 'Holiday', 'Wind speed (m/s)']
        df = df.drop(columns=columns_to_drop)
        
        # Escalar las características
        X = scaler.transform(df)
        
        # Predicción
        predictions = model.predict(X)
        logger.info(f"Predictions: {predictions}")
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)})

@app.route("/train", methods=["GET"])
def train():
    if train_status == "training":
        return jsonify({"error": "Training already in progress"})
    _train()
    return jsonify({"result": "Training finished successfully"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
