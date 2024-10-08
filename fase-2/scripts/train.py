# train.py

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import subprocess

# Descargar los datos desde Google Drive
subprocess.run(['wget', '--no-check-certificate', 'https://drive.google.com/uc?id=1gt4gMHp6cW4SUDWOcnsRzS8KLyx4CaY5', '-O', 'train.csv'], check=True)
# Carga del archivo CSV con formato explícito
df_train = pd.read_csv('train.csv', encoding='unicode_escape', parse_dates=['Date'], dayfirst=True)

# Forzar la conversión a formato de fecha si la advertencia persiste
df_train['Date'] = pd.to_datetime(df_train['Date'], format='%d/%m/%Y', errors='coerce')


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

print("Iniciando el entrenamiento...")

# Agregar columnas necesarias
df_train['Month'] = pd.DatetimeIndex(df_train['Date']).month
df_train['Day'] = pd.DatetimeIndex(df_train['Date']).day
df_train['Weekday'] = pd.DatetimeIndex(df_train['Date']).weekday

# Preprocesar los datos
df_train = pre_processing(df_train)
df_train = add_working_hour_column(df_train)

print("2do mensaje entrenamiento...")

# Preparar los datos para el entrenamiento
X = df_train.drop(columns=['y', 'Month', 'Day'])
y = df_train['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3er mensaje entrenamiento...")

# Estandarización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo XGBoost
XGBModel = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.6, gamma=2, max_depth=6, subsample=0.7, reg_alpha=0.15, reg_lambda=1, learning_rate=0.15)
XGBModel.fit(X_train, y_train)


# Verificar si el scaler ha sido ajustado
if not hasattr(scaler, 'mean_'):
    raise ValueError("El StandardScaler no ha sido ajustado correctamente.")


# Guardar el modelo y el scaler

joblib.dump(XGBModel, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Scaler guardado exitosamente en scaler.pkl")

# Evaluación
y_pred = XGBModel.predict(X_test)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f'RMSLE: {rmsle}')

# Visualización
sns.histplot(y_train, bins=20)
sns.histplot(y_test, bins=20)
sns.histplot(y_pred, bins=20)
plt.show()
print("Entrenamiento completado y modelos guardados exitosamente.")


