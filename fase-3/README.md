**FASE 3. Despliegue de API Rest**

Para el despliegue del contenedor del modelo de prediccion del sistema de bicicletas de Seul se deben seguir los siguientes pasos:

Instalar Doker, y que este se encuentre en estado activo.

Clonar el repositorio (se recomienta utilizar Visual code para su manejo).

Acceder a la carpeta de fase 3 desde terminal (cd fase-3).

Construir la imagen: Para esto ejecutamos: "docker build -t imagen-modelo-bicicletas .".

Ejecutar el contenedor: Se ejecuta: "docker run -d --name contenedor-api-bicicletas -p 5000:5000 api-modelo-bicicletas".

Para ejecutar la imagen se puede usar la siguiente linea: "docker run -it --rm imagen-modelo-bicicletas".

Los endpoint se abren mediante url: curl -X GET http://localhost:5000/status curl -X GET http://localhost:5000/train curl -X GET http://localhost:5000/predict
