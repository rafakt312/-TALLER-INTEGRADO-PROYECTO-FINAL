from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
# IMPORTANTE: Importar StaticFiles para servir imágenes
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import io
import os
from collections import Counter
from sklearn.metrics import confusion_matrix

app = FastAPI(title="Examen HAR - Pipeline Completo", version="Final.Fixed.Img")

# --- CONFIGURACIÓN DE ARCHIVOS ESTÁTICOS ---
# ⚠️ DEBES CREAR UNA CARPETA LLAMADA 'static' JUNTO A ESTE script
# Y COLOCAR AHÍ TU IMAGEN 'image_8.png'
app.mount("/static", StaticFiles(directory="static"), name="static")

# Rutas y Configuración
MODEL_PATH = "models/mhealth_rf_model.pkl"
FS = 50
WINDOW_SECONDS = 2
WINDOW_SIZE = FS * WINDOW_SECONDS
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

ACTIVITY_LABELS = {
    0: "Null", 1: "L1: De pie", 2: "L2: Sentado", 3: "L3: Acostado", 4: "L4: Caminar",
    5: "L5: Subir escaleras", 6: "L6: Flex. cintura", 7: "L7: Brazos arriba", 8: "L8: Agacharse",
    9: "L9: Ciclismo", 10: "L10: Trotar", 11: "L11: Correr", 12: "L12: Saltar"
}

model = None

# --- HTML FRONTEND ---
html_content = r"""
<!DOCTYPE html>
<html>
<head>
    <title>Examen HAR - Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f8f9fa; color: #333; }
        .container { max-width: 1300px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        
        .control-panel { background: #e9ecef; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 30px; }
        button { background-color: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold; }
        button:hover { background-color: #0056b3; }

        .section-title { border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-top: 40px; margin-bottom: 20px; color: #495057; font-size: 1.5em; }
        
        /* Estilo para la imagen estática */
        .static-image-container {
            text-align: center;
            margin-bottom: 30px;
        }
        .static-image {
            max-width: 100%;
            height: auto;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }

        .matrix-wrapper { display: flex; justify-content: center; overflow-x: auto; margin-top: 20px; }
        table { border-collapse: collapse; font-size: 0.9em; }
        th, td { padding: 8px; text-align: center; border: 1px solid #dee2e6; }
        th { background-color: #f1f3f5; }
        .cell-ok { background-color: #d4edda; color: #155724; font-weight: bold; }
        .cell-err { background-color: #f8d7da; color: #721c24; font-weight: bold; }
        .cell-zero { color: #eee; }

        .loading { display: none; font-weight: bold; color: #666; margin-top: 10px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Examen: Reconocimiento de Actividad Humana</h1>
        <div class="control-panel">
            <p>Seleccione archivo de validación (Sujeto 7 u 8):</p>
            <input type="file" id="logFile" accept=".log,.txt">
            <br><br>
            <button onclick="procesar()">Analizar Archivo</button>
            <p id="loading" class="loading">Procesando modelo IA...</p>
        </div>

        <div id="resultSection" class="hidden">
            <h2 class="section-title">1. Matriz de Confusión (Imagen Estática)</h2>
            <div class="static-image-container">
                <img src="/static/image_8.png" alt="Matriz de Confusión del Modelo" class="static-image">
            </div>

            <h2 class="section-title">2. Matriz de Confusión (Calculada del Archivo Actual)</h2>
            <p style="text-align: center; color: #666;">(Esta matriz se calcula dinámicamente con los datos del archivo que acabas de subir)</p>
            <div class="matrix-wrapper"><table id="confMatrix"></table></div>
        </div>
    </div>

    <script>
        // YA NO NECESITAMOS GOOGLE CHARTS PARA TIMELINE
        // google.charts.load('current', {'packages':['timeline']});

        async function procesar() {
            const file = document.getElementById('logFile').files[0];
            if (!file) { alert("Seleccione un archivo"); return; }
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultSection').classList.add('hidden');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                
                // YA NO DIBUJAMOS LA LÍNEA DE TIEMPO
                // drawTimeline(data.timeline_pred, data.timeline_true);
                
                if(data.confusion_matrix && data.confusion_matrix.length > 0) {
                    drawMatrix(data.confusion_matrix, data.labels);
                }
                document.getElementById('resultSection').classList.remove('hidden');
            } catch (e) { alert("Error: " + e.message); } 
            finally { document.getElementById('loading').style.display = 'none'; }
        }

        // FUNCIÓN drawTimeline ELIMINADA

        function drawMatrix(matrix, labels) {
            const table = document.getElementById("confMatrix");
            table.innerHTML = "";
            let head = "<thead><tr><th>Real \\ Pred</th>";
            labels.forEach(l => head += `<th>${l.split(':')[0]}</th>`);
            head += "</tr></thead>";
            table.innerHTML += head;
            let body = "<tbody>";
            matrix.forEach((row, i) => {
                body += `<tr><th>${labels[i].split(':')[0]}</th>`;
                row.forEach((val, j) => {
                    const pct = (val * 100).toFixed(0);
                    let cls = "cell-zero"; let txt = "";
                    if (pct > 0) { txt = pct + "%"; cls = (i === j) ? "cell-ok" : "cell-err"; }
                    body += `<td class="${cls}">${txt}</td>`;
                });
                body += "</tr>";
            });
            table.innerHTML += body + "</tbody>";
        }
    </script>
</body>
</html>
"""

# --- BACKEND LOGIC ---
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ Modelo PKL cargado correctamente.")
        except Exception as e:
            print(f"⚠️ Advertencia de versión al cargar PKL: {e}")
            print("Intentando cargar de todas formas...")
    else:
        print("❌ Error: No se encuentra el modelo. Ejecuta pipeline_mhealth.py primero.")

@app.get("/", response_class=HTMLResponse)
def index(): return html_content

def calc_mag(df, c1, c2, c3):
    return np.sqrt(df[c1]**2 + df[c2]**2 + df[c3]**2)

def extraer_estadisticas(ventana):
    mean = np.mean(ventana, axis=0)
    std = np.std(ventana, axis=0)
    max_val = np.max(ventana, axis=0)
    min_val = np.min(ventana, axis=0)
    median = np.median(ventana, axis=0)
    ptp = np.ptp(ventana, axis=0)
    var = np.var(ventana, axis=0)
    return np.concatenate([mean, std, max_val, min_val, median, ptp, var])

# Funciones de suavizado y timeline eliminadas del backend pues no se usan visualmente

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None: raise HTTPException(500, "Modelo no cargado")
    try:
        content = await file.read()
        try: df = pd.read_csv(io.BytesIO(content), sep='\t', header=None)
        except: df = pd.read_csv(io.BytesIO(content), delim_whitespace=True, header=None)

        # Definir Columnas RAW MHEALTH
        col_raw = [
            "acc_ch_x", "acc_ch_y", "acc_ch_z", "ecg_1", "ecg_2",
            "acc_ank_x", "acc_ank_y", "acc_ank_z", "gyro_ank_x", "gyro_ank_y", "gyro_ank_z", "mag_ank_x", "mag_ank_y", "mag_ank_z",
            "acc_arm_x", "acc_arm_y", "acc_arm_z", "gyro_arm_x", "gyro_arm_y", "gyro_arm_z", "mag_arm_x", "mag_arm_y", "mag_arm_z",
            "label"
        ]
        df.columns = col_raw
        
        # Ingeniería de Features (Expandir a 26 canales)
        df['mag_ch'] = calc_mag(df, 'acc_ch_x', 'acc_ch_y', 'acc_ch_z')
        df['mag_ank'] = calc_mag(df, 'acc_ank_x', 'acc_ank_y', 'acc_ank_z')
        df['mag_arm'] = calc_mag(df, 'acc_arm_x', 'acc_arm_y', 'acc_arm_z')
        
        # Reordenar para que coincida con el entrenamiento
        cols_features = [c for c in df.columns if c != 'label']
        df_feat = df[cols_features] # 26 canales
        
        preds_list = []; true_list = []
        
        for start in range(0, len(df) - int(WINDOW_SIZE), STEP_SIZE):
            end = start + int(WINDOW_SIZE)
            win_data = df_feat.iloc[start:end].values
            
            f = extraer_estadisticas(win_data).reshape(1, -1)
            
            # Predecir
            lbl = model.predict(f)[0]
            preds_list.append(ACTIVITY_LABELS.get(lbl, str(lbl)))
            
            # Etiqueta Real
            try:
                real = int(df.iloc[start:end]['label'].mode()[0])
                true_list.append(real)
            except: true_list.append(0)

        # Solo calculamos la matriz de confusión dinámica
        mtx = []
        labels_mtx = []
        if true_list:
            pred_ids = [k for k,v in ACTIVITY_LABELS.items() if v in preds_list]
            all_ids = sorted(list(set(true_list) | set(pred_ids)))
            if 0 in all_ids: all_ids.remove(0)
            
            labels_mtx = [ACTIVITY_LABELS.get(i, str(i)) for i in all_ids]
            rev_map = {v:k for k,v in ACTIVITY_LABELS.items()}
            y_pred_ids = [rev_map.get(p,0) for p in preds_list]
            
            y_true_clean = []
            y_pred_clean = []
            for t, p in zip(true_list, y_pred_ids):
                if t != 0:
                    y_true_clean.append(t)
                    y_pred_clean.append(p)
            
            if y_true_clean:
                cm = confusion_matrix(y_true_clean, y_pred_clean, labels=all_ids, normalize='true')
                mtx = cm.tolist()

        return {
            # Ya no enviamos datos de timeline
            "confusion_matrix": mtx,
            "labels": labels_mtx
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}