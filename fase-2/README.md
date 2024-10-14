FASE 2. Despliegue en container

Para el despliegue del contenedor del modelo de prediccion del sistema de bicicletas de Seul se deben seguir los siguientes pasos:

1. Instalar Doker, y que este se encuentre en estado activo.

2. Clonar el repositorio (se recomienta utilizar Visual code para su manejo).

2. Acceder a la carpeta de fase 2 desde terminal (cd fase-2).

4. Construir la imagen:
	Para esto ejecutamos: "docker build -t imagen-modelo-bicicletas .".

5. Ejecutar el contenedor:
	Se ejecuta: "docker run -d --name contenedor-modelo-bicicletas imagen-modelo-bicicletas".

Para ejecutar la imagen se puede usar la siguiente linea: "docker run -it --rm imagen-modelo-bicicletas".

Se ha creado el archivo de predicciones en "submission.csv" dentro de los archivos del contenedor.
Para exportarlo a su maquina puede usar el siguiente comando: "docker cp <container_id>:/app/path_to_output_file ./output_filce" donde se debe reemplazar container_id con la id del contenedor creado anteriormente (este se puede obtener al ejecutar: " docker ps"), y la direccion donde se desea traer.
