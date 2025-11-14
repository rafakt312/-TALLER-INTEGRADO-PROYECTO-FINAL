# Carpeta de Prompts: Colaboración con IA Generativa (Gemini)

Este archivo documenta el uso de IA generativa, como se solicita en la Etapa 1, punto 6, de las instrucciones del proyecto. Se utilizó el modelo Gemini de Google para asistir en el desarrollo, depuración y estructuración de la aplicación.

El proceso fue iterativo y se centró en la arquitectura del despliegue más que en la generación del modelo (que se completó en el Producto 3).

## Prompts y Tareas Clave

A continuación, se resumen los "prompts" (instrucciones) más relevantes que se usaron para construir la aplicación:

### 1. Diseño de Arquitectura (Back End + Front End)
* **Prompt/Tarea:** "Quiero usar Python para el Back End, pero no sé qué usar para el Front End."
* **Asistencia:** Se discutieron las opciones (Vanilla JS vs. React/Vue) y se recomendó Vanilla JS (HTML/CSS/JS puros) para cumplir los requisitos del proyecto sin añadir complejidad innecesaria.

### 2. Creación del Esqueleto del Back End (FastAPI)
* **Prompt/Tarea:** "Sigue adelante [con la recomendación de FastAPI]."
* **Asistencia:** Se generó el código base de `main.py` (Etapa 1), incluyendo:
    * Endpoint `/health`.
    * Endpoint `/detect` simulado (mockup) que acepta la subida de archivos.
    * Lógica para generar y devolver un gráfico de Matplotlib codificado en Base64.

### 3. Creación del Esqueleto del Front End (HTML/JS)
* **Prompt/Tarea:** (Continuación de la creación del Back End).
* **Asistencia:** Se generó el archivo `index.html` completo, incluyendo:
    * El formulario de subida.
    * El código JavaScript (con `fetch`) para llamar al endpoint `/detect` del Back End.
    * La lógica para manejar la respuesta JSON y mostrar el resultado y el gráfico.

### 4. Depuración de CORS
* **Prompt/Tarea:** "Me da un error de 'No se pudo conectar con el servidor'."
* **Asistencia:** Se diagnosticó el problema como un error de **CORS** (Cross-Origin Resource Sharing) y se proporcionó el código exacto del `CORSMiddleware` para solucionarlo en `main.py`.

### 5. Contenerización con Docker
* **Prompt/Tarea:** "¿Por qué me dicen que es bueno usar Docker en este proyecto?"
* **Asistencia:** Se explicó el concepto de **reproducibilidad** y se generó la arquitectura de **Docker Compose** (Etapa 1 y 4.3):
    * `Dockerfile` para el servicio de Back End (Python/FastAPI).
    * `docker-compose.yml` para orquestar dos servicios: `backend` (FastAPI) y `frontend` (Nginx sirviendo el `index.html`).
    * `requirements.txt` para manejar las dependencias de Python.

### 6. Depuración de Docker
* **Prompt/Tarea:** (Usuario pega un error de `contourpy` y `python:3.10`).
* **Asistencia:** Se diagnosticó el conflicto de versiones entre el `requirements.txt` (que pedía Python 3.11+) y el `Dockerfile` (que usaba Python 3.10). Se corrigió la imagen base a `python:3.11-slim`.

### 7. Integración del Modelo Real (Pipeline)
* **Prompt/Tarea:** (Usuario envía el notebook "Producto 3").
* **Asistencia:** Se analizó el notebook `Producto3_MHealth_Modelos_GonzalezUrbina.ipynb` para extraer la lógica de pre-procesamiento exacta.
* **Prompt/Tarea:** "No tengo el archivo del modelo."
* **Asistencia:** Se identificó que el modelo `rf_model` nunca se guardó (serializó). Se proporcionó el código (`joblib.dump(rf_model, "rf_model.joblib")`) para guardar el modelo entrenado.
* **Asistencia:** Se actualizó el `main.py` final, reemplazando la simulación por el pipeline real:
    * Carga del modelo `rf_model.joblib` al inicio.
    * Función `process_log_file` que replica la limpieza y selección de features del notebook.
    * Lógica para predecir y encontrar la actividad más frecuente (moda) del archivo.