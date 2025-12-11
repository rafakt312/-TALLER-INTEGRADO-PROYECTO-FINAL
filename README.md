# Proyecto Final: Aplicación Web de Reconocimiento de Actividad MHealth

Esta es la entrega del Proyecto Final del Taller Integrado. El objetivo es una aplicación web completa que despliega un modelo de Machine Learning (Random Forest) entrenado para reconocer actividades humanas (ej. "Caminando", "Sentado") a partir de datos de sensores del dataset MHealth.

El énfasis del proyecto está en el **despliegue técnico** y la integración de todo el ciclo de vida del software (Datos -> Modelo -> API -> Interfaz de Usuario), cumpliendo con todos los requisitos de la Etapa 1.

## 🚀 Arquitectura del Sistema

La aplicación sigue una arquitectura de microservicios, gestionada íntegramente por **Docker Compose**. Esto garantiza un despliegue **reproducible** y consistente.

La arquitectura consta de dos servicios principales:

* **Servicio `backend` (Python/FastAPI):**
    * Es una API de **FastAPI** construida sobre una imagen de **Python 3.11**.
    * Carga el modelo `rf_model.joblib` (un Random Forest entrenado en el "Producto 3") al iniciarse.
    * Expone el endpoint `POST /detect` que recibe un archivo `.log`.
    * Aplica el pipeline de pre-procesamiento (limpieza de datos y selección de 21 features) idéntico al del notebook de entrenamiento.
    * Devuelve la predicción de la actividad más frecuente (moda) y un gráfico de muestra en formato JSON.

* **Servicio `frontend` (Nginx/HTML):**
    * Es un servidor web **Nginx** (Alpine) que sirve un único archivo `index.html`.
    * El `index.html` contiene **JavaScript "puro" (Vanilla JS)** que se encarga de:
        * Mostrar el formulario de subida.
        * Llamar (vía `fetch`) al endpoint `/detect` del `backend` cuando el usuario sube un archivo.
        * Renderizar la respuesta (actividad y gráfico) en la página.

## 🛠️ Tecnologías Utilizadas

* **Backend:** Python 3.11, FastAPI, Pandas, Scikit-learn, Joblib, Matplotlib
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla JS)
* **Servidor Web (Frontend):** Nginx
* **Despliegue y Orquestación:** Docker & Docker Compose

## 📋 Prerrequisitos

Para ejecutar este proyecto, solo necesitas tener una dependencia instalada en tu máquina:

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Debe estar en ejecución)

## ⚡ Instrucciones de Despliegue y Uso

El proyecto está 100% contenerizado. No es necesario instalar Python, `pip`, `venv` ni Nginx localmente. Docker se encarga de todo.

1.  Clona o descarga este repositorio en tu máquina.
2.  Abre una terminal en la carpeta raíz del proyecto (donde se encuentra el archivo `docker-compose.yml`).
3.  Ejecuta el siguiente comando. Esto construirá las imágenes de Docker (la primera vez puede tardar unos minutos) y levantará ambos servicios:

    ```bash
    docker-compose up --build
    ```

4.  Espera a que la terminal termine de construir y muestre los logs de los servicios `backend-1` y `frontend-1`, indicando que están en funcionamiento.

### Cómo Probar la Aplicación

1.  Una vez que los contenedores estén corriendo, abre tu navegador web y ve a:

    **[http://localhost](http://localhost)**
    *(Nota: Es `http://localhost`, no `localhost:8000`)*

2.  Verás la interfaz "Detector de Actividad MHealth".
3.  Usa el formulario para subir uno de los archivos `.log` del dataset MHealth (ej. `mHealth_subject1.log`).
4.  Presiona el botón "Analizar Actividad".
5.  El sistema contactará al `backend`, procesará el archivo y mostrará la predicción del modelo en tiempo real.

### Para Detener la Aplicación

* Vuelve a la terminal donde ejecutaste `docker-compose up` y presiona `CTRL + C`.

Nota sobre el Modelo: El archivo rf_model.joblib no está incluido en este repositorio debido a su tamaño. Para ejecutar el proyecto, se debe generar ejecutando el notebook Producto3.ipynb o solicitar el archivo directamente al autor. Debe colocarse en la misma carpeta que main.py.