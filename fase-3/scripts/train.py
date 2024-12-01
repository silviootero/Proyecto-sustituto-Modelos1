# train.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import gdown
from sklearn.preprocessing import StandardScaler

# Funciones de preprocesamiento
def pre_processing(df):
    df["Seasons"] = df["Seasons"].astype("category").cat.codes
    df["Functioning Day"] = df["Functioning Day"].astype("category").cat.codes
    df["Holiday"] = df["Holiday"].astype("category").cat.codes
    columns_to_drop = ['Date', 'Snowfall (cm)', 'Holiday', 'Wind speed (m/s)']
    df = df.drop(columns=columns_to_drop)
    return df

def add_working_hour_column(df):
    df["working_hour"] = ((df["Hour"] >= 5) & (df["Hour"] <= 20)).astype(int)
    return df

# URL del archivo de Google Drive
url = 'https://drive.google.com/uc?id=1gt4gMHp6cW4SUDWOcnsRzS8KLyx4CaY5'
output = 'train.csv'
gdown.download(url, output, quiet=False)

# Cargar los datos
df_train = pd.read_csv('train.csv', encoding='unicode_escape')

# Convertir 'Date' a datetime y extraer componentes
df_train['Date'] = pd.to_datetime(df_train['Date'], format='%d/%m/%Y')
df_train['Month'] = df_train['Date'].dt.month
df_train['Day'] = df_train['Date'].dt.day
df_train['Weekday'] = df_train['Date'].dt.weekday

# Renombrar columnas para facilitar su uso
df_train = df_train.rename(columns={
    'Temperature(°C)': 'Temperature (C)',
    'Dew point temperature(°C)': 'Dew point temperature (C)'
})

# Aplicar preprocesamiento y añadir columna de horas laborales
df_train = pre_processing(df_train)
df_train = add_working_hour_column(df_train)

# Reemplazo de valores atípicos (outliers)
def replace_outliers(df):
    for feature in df.drop(columns=["Hour", "Month", "Day", 'Functioning Day', 'Seasons']).columns:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        upperL = Q3 + 1.5 * IQR
        lowerL = Q1 - 1.5 * IQR
        df[feature] = df[feature].clip(lower=lowerL, upper=upperL)
    return df

df_train = replace_outliers(df_train)

# Separar las características (X) y la variable objetivo (y)
X = df_train.drop(columns=['y'])
y = df_train['y']

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo
model = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.6, gamma=2, max_depth=6, subsample=.7, reg_alpha=0.15, reg_lambda=1, learning_rate=0.15)
model.fit(X_train, y_train)

# Guardar el modelo y el escalador
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Modelo y escalador guardados exitosamente.")

