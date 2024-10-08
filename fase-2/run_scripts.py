import subprocess
import sys

def run_script(script_name):
    """Función para ejecutar un script de Python dado su nombre"""
    try:
        print(f"Ejecutando {script_name}...")
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(f"Salida de {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar {script_name}:\n{e.stderr}")
        exit(1)

if __name__ == "__main__":
    # Paso 1: Ejecutar el script de entrenamiento
    run_script('scripts/train.py')

    # Paso 2: Ejecutar el script de predicción
    run_script('scripts/predict.py')

    print("Entrenamiento y predicción completados con éxito.")
