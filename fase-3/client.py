import requests

# Endpoint para predicción
url = 'http://localhost:5000/predict'

# Ejemplo de datos a enviar
data = [
    {
        "Hour": 14,
        "Temperature (C)": 23.0,
        "Dew point temperature (C)": 16.0,
        "Solar Radiation (MJ/m2)": 2.5,
        "Rainfall (mm)": 0.0,
        "Seasons": "Autumn",
        "Functioning Day": "Yes"
    }
]

response = requests.post(url, json=data)
print("Response from API:", response.json())

