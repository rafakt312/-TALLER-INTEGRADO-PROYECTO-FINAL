# 1. Imagen base ligera de Python
FROM python:3.9-slim

# 2. Evitar archivos temporales de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Crear directorio de trabajo
WORKDIR /app

# 4. Instalar librerías del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 5. Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar el resto del código
COPY . .

# 7. Exponer el puerto
EXPOSE 8000

# 8. Ejecutar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]