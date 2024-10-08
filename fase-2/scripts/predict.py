# predict.py

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
import subprocess

# Descargar los datos de prueba desde Google Drive
# Descargar los datos desde Google Drive
subprocess.run(['wget', '--no-check-certificate', 'https://drive.google.com/uc?id=1m1Abp4lseZDXUsDlEjKmxcv1UHDmGDZY', '-O', 'test.csv'], check=True)
df_test = pd.read_csv('test.csv', encoding='unicode_escape')

# Eliminar la columna 'y' en test, ya que es el objetivo a predecir
df_test = df_test.drop(columns=['y'])

# Preprocesamiento
def add_working_hour_column(df):
    df["working_hour"] = 0
    df["working_hour"] = ((df["Hour"] >= 5) & (df["Hour"] <= 20)).astype(int)
    return df

def encode_categroical_features(df):
    df["Seasons"] = df["Seasons"].astype("category").cat.codes
    df["Functioning Day"] = df["Functioning Day"].astype("category").cat.codes
    df["Holiday"] = df["Holiday"].astype("category").cat.codes
    return df

def pre_processing(df):
    columns_to_drop = ['Date', 'Snowfall (cm)', 'Holiday', 'Wind speed (m/s)']
    df = encode_categroical_features(df)
    df = df.drop(columns=columns_to_drop)
    return df

# Agregar columnas necesarias
df_test['Month'] = pd.DatetimeIndex(df_test['Date']).month
df_test['Day'] = pd.DatetimeIndex(df_test['Date']).day
df_test['Weekday'] = pd.DatetimeIndex(df_test['Date']).weekday

# Preprocesar los datos
df_test = pre_processing(df_test)
df_test = add_working_hour_column(df_test)

print("Archivos en el directorio actual:", os.listdir())

# Cargar el modelo y el scaler entrenado
XGBModel = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')


# Estandarizar los datos de prueba
X_test = df_test.drop(columns=['Month', 'Day'])
X_test = scaler.transform(X_test)

# Realizar predicciones
y_test_predicted = XGBModel.predict(X_test)

# Guardar los resultados
df_test['y'] = y_test_predicted
df_test[['y']].to_csv('submission.csv', index=False)

print(df_test['y'])
