from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np

# --- 1. DEFINICIONES GLOBALES (De tu Notebook "Producto 3") ---

# (De 1.2) Nombres de las 24 columnas del archivo .log
COLUMN_NAMES = [
    'Chest_Accel_X', 'Chest_Accel_Y', 'Chest_Accel_Z',
    'Chest_ECG_Lead1', 'Chest_ECG_Lead2',
    'Ankle_Accel_X', 'Ankle_Accel_Y', 'Ankle_Accel_Z',
    'Ankle_Gyro_X', 'Ankle_Gyro_Y', 'Ankle_Gyro_Z',
    'Ankle_Mag_X', 'Ankle_Mag_Y', 'Ankle_Mag_Z',
    'Arm_Accel_X', 'Arm_Accel_Y', 'Arm_Accel_Z',
    'Arm_Gyro_X', 'Arm_Gyro_Y', 'Arm_Gyro_Z',
    'Arm_Mag_X', 'Arm_Mag_Y', 'Arm_Mag_Z',
    'Label'
]

# (De 1.3) Las 21 características (features) que usaste para entrenar
FEATURES_LIST = [
    'Chest_Accel_X', 'Chest_Accel_Y', 'Chest_Accel_Z',
    'Ankle_Accel_X', 'Ankle_Accel_Y', 'Ankle_Accel_Z',
    'Ankle_Gyro_X', 'Ankle_Gyro_Y', 'Ankle_Gyro_Z',
    'Ankle_Mag_X', 'Ankle_Mag_Y', 'Ankle_Mag_Z',
    'Arm_Accel_X', 'Arm_Accel_Y', 'Arm_Accel_Z',
    'Arm_Gyro_X', 'Arm_Gyro_Y', 'Arm_Gyro_Z',
    'Arm_Mag_X', 'Arm_Mag_Y', 'Arm_Mag_Z'
]

# (De 1.2) Mapeo de ID de actividad a Nombre
ACTIVITY_LABELS = {
    1: 'Standing', 2: 'Sitting', 3: 'Lying',
    4: 'Walking', 5: 'Climbing Stairs', 6: 'Waist Bends',
    7: 'Arm Elevation', 8: 'Knees Bending', 9: 'Cycling',
    10: 'Jogging', 11: 'Running', 12: 'Jump Front & Back'
}

# --- 2. CONFIGURACIÓN DE APP Y CORS ---
app = FastAPI()
origins = ["http://localhost", "http://127.0.0.1"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. CARGAR EL MODELO AL INICIO ---
# Carga tu modelo .joblib o .pkl aquí.
MODEL_PATH = "rf_model.joblib" # <-- ¡CONFIRMA ESTE NOMBRE!
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo '{MODEL_PATH}' cargado exitosamente.")
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de modelo en '{MODEL_PATH}'.")
    model = None
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo: {e}")
    model = None


# --- 4. FUNCIÓN DEL PIPELINE (De tu Taller) ---
def process_log_file(raw_bytes: bytes):
    """
    Toma los bytes crudos de un archivo .log y aplica el pipeline
    de tu "Producto 3" para prepararlo para el modelo RF.
    """
    # 1. (De 1.2) Cargar los bytes en un DataFrame de Pandas
    # Usamos io.BytesIO para que pandas lea los bytes como si fueran un archivo
    string_data = io.BytesIO(raw_bytes)
    df_subject = pd.read_csv(string_data, sep='\s+', header=None, names=COLUMN_NAMES)
    
    # 2. (De 1.3) Limpieza: Eliminar la clase 0 (Nula)
    # Hacemos esto por si el archivo de entrada las incluye
    df_clean = df_subject[df_subject['Label'] != 0].copy()
    
    # 3. (De 1.3) Selección de Variables: Extraer solo las 21 features
    # (¡No necesitamos escalar, porque es Random Forest!)
    X_new = df_clean[FEATURES_LIST]
    
    return X_new

# --- 5. ENDPOINTS DE LA API ---

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Servicio de detección MHealth funcionando."}

@app.post("/detect")
async def detect_activity(file: UploadFile = File(...)):
    
    if model is None:
        return JSONResponse(
            status_code=500, 
            content={"error": "Modelo no cargado en el servidor."}
        )
    
    # 1. Leer los bytes crudos del archivo subido
    raw_bytes = await file.read()
    
    try:
        # 2. Aplicar el pipeline de procesamiento (Paso 4 de este script)
        datos_listos_para_modelo = process_log_file(raw_bytes)
        
        if datos_listos_para_modelo.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "No se encontraron datos válidos (Label != 0) en el archivo."}
            )

        # 3. Ejecutar el modelo (RF)
        # El modelo predecirá CADA fila (timestamp) del archivo
        predictions = model.predict(datos_listos_para_modelo)
        
        # 4. Post-procesar la salida:
        #   Un archivo .log tiene muchas actividades. Devolveremos
        #   la actividad más común (la moda) en el archivo.
        most_frequent_prediction_id = pd.Series(predictions).mode()[0]
        actividad_texto = ACTIVITY_LABELS.get(int(most_frequent_prediction_id), "ID Desconocido")

        # 5. Generar el gráfico (usando el 1er sensor como ejemplo)
        fig, ax = plt.subplots(figsize=(10, 4))
        # Graficar los primeros 500 puntos para no saturar
        plot_data = datos_listos_para_modelo['Chest_Accel_X'].head(500)
        ax.plot(plot_data)
        ax.set_title(f"Muestra de 'Chest_Accel_X' ({file.filename})")
        ax.set_ylabel("Aceleración")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        image_base64_string = f"data:image/png;base64,{image_base64}"
        plt.close(fig)

        # 6. Devolver la respuesta
        response_data = {
            "actividad_detectada": f"{actividad_texto} (ID: {most_frequent_prediction_id})",
            "info_adicional": f"Actividad más frecuente detectada en el archivo {file.filename}.",
            "grafico_base64": image_base64_string
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        # Captura cualquier error durante el pipeline o la predicción
        return JSONResponse(
            status_code=400, 
            content={"error": f"No se pudo procesar el archivo: {str(e)}"}
        )

@app.get("/")
def read_root():
    return {"Hola": "Visita /docs para ver la documentación de la API."}