# 1. Imagen base: Usamos una imagen oficial de Python 3.11
FROM python:3.11-slim
# 2. Directorio de trabajo: Creamos una carpeta /app dentro del contenedor
WORKDIR /app

# 3. Copiar la lista de dependencias
COPY requirements.txt .

# 4. Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar todo el código de tu proyecto (main.py, etc.) a la carpeta /app
COPY . .

# 6. Comando de ejecución: Le decimos a Docker cómo correr tu app
#    (Usamos 0.0.0.0 para que sea accesible desde fuera del contenedor)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]