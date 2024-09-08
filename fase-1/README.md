**DESARROLLO DEL NOTEBOOK PASO A PASO**

La competencia tomada para la realización de este proyecto es Seoul Bike Rental Prediction - AI-Pro - ITI https://www.kaggle.com/competitions/seoul-bike-rental-ai-pro-iti/overview

Se desarrolla en un notebook en el que se va abordando paso a paso de la siguiente manera:

• **Importación de librerías:** Se importan bibliotecas clave como numpy, pandas, matplotlib, seaborn y herramientas de scikit-learn para la manipulación de datos, visualización y modelado. Además, se importa stats de scipy para análisis estadístico y PCA para reducción de dimensionalidad.

• **Carga de los datos:** Se cargan dos archivos CSV (train.csv y test.csv) en dataframes de pandas. Luego, se elimina la columna 'y' del conjunto de prueba ya que representa el valor a predecir.

• **Inspección de los datos:** Se observa la estructura de los datos mediante head(), se listan las columnas y sus tipos de datos, y se calcula un resumen estadístico de las columnas con describe().

• **Limpieza de datos:** Se renombraron algunas columnas como "Temperature (C)" y "Dew point temperature (C)" para mayor claridad. Además, se crea una función para agregar una columna indicando si las horas son laborales (entre las 5 y las 20 horas). También, se separa la columna de fecha en mes, día y día de la semana.


• **Procesamiento de características:**

    • Se codifican las características categóricas como Seasons, Functioning Day, y Holiday.
    
    •	Se aplica PCA para reducir las dimensiones de las características de temperatura.
    
    •	Se agrega la función para filtrar los días de funcionamiento.
    
    •	Se reemplazan los valores atípicos en los datos numéricos utilizando el método IQR para eliminar valores extremos.

    
    
• **Preprocesamiento:** Se eliminan columnas no relevantes como la fecha, Snowfall (cm), Holiday y Wind speed (m/s). Luego, se normalizan las características numéricas para estandarizar los datos de entrenamiento y prueba.


• **Modelado:**

    •	Se utiliza el modelo XGBRegressor de XGBoost para la predicción, ajustando varios hiperparámetros como tweedie_variance_power, max_depth, y learning_rate. El conjunto de entrenamiento se ajusta y el modelo se entrena con una división train_test_split para evaluar su rendimiento.
    
    •	Se calcula el error con mean_squared_log_error y se evalúa la precisión del modelo.

    
• **Visualización:** Se generan gráficos como histogramas y diagramas de dispersión para visualizar la distribución de los valores de predicción y los errores entre los valores reales y predichos.

• **Predicción final:** Después de preprocesar el conjunto de prueba, se realiza la predicción final y se guarda en un archivo CSV.
