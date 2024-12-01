import pandas as pd
import joblib
import gdown

# Funciones de preprocesamiento (mismas que en train.py)
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


# Cargar el modelo y el escalador
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# URL del archivo de Google Drive
url = 'https://drive.google.com/uc?id=1cXRKINwgO83RcymHp-VdJhQfRv9mXIEZ'
output = 'test.csv'

gdown.download(url, output, quiet=False)

# Cargar el conjunto de datos para predicción
df_test = pd.read_csv('test.csv', encoding='unicode_escape')

# Convertir 'Date' a datetime y extraer componentes
df_test['Date'] = pd.to_datetime(df_test['Date'], format='%d/%m/%Y')
df_test['Month'] = df_test['Date'].dt.month
df_test['Day'] = df_test['Date'].dt.day
df_test['Weekday'] = df_test['Date'].dt.weekday

# Renombrar columnas para facilitar su uso
df_test.rename(columns={
    'Dew point temperature(Â°C)': 'Dew point temperature (C)',
    'Temperature(Â°C)': 'Temperature (C)'
}, inplace=True)

# Aplicar preprocesamiento y añadir columna de horas laborales
df_test = pre_processing(df_test)
df_test = add_working_hour_column(df_test)

# Preprocesar el conjunto de datos de prueba
X_test = df_test.drop(columns=['y'])  # Asegurarse de que la columna 'y' no esté incluida si existe
X_test = scaler.transform(X_test)

# Realizar predicciones
y_pred = model.predict(X_test)
df_test['y'] = y_pred

# Guardar las predicciones en un archivo CSV
df_test[['y']].to_csv('submission.csv', index=False)

print("Predicciones guardadas en submission.csv.")
