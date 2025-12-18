import os
import urllib.request
import zipfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy import stats

# ==========================================
# 1. CONFIGURACIÓN GLOBAL
# ==========================================
DATASET_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"
MODELS_DIR = "models"
OUTPUT_DIR = "output"

# Configuración Física MHEALTH
FS = 50  # Frecuencia de muestreo (Hz)
WINDOW_SECONDS = 2
WINDOW_SIZE = FS * WINDOW_SECONDS  # 100 muestras
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP)) # 50 muestras

# Definición de Canales (23 Originales + 3 Magnitudes = 26)
COLUMNAS_RAW = [
    "acc_ch_x", "acc_ch_y", "acc_ch_z",       # 0-2: Pecho
    "ecg_1", "ecg_2",                         # 3-4: ECG
    "acc_ank_x", "acc_ank_y", "acc_ank_z",    # 5-7: Tobillo Acc
    "gyro_ank_x", "gyro_ank_y", "gyro_ank_z", # 8-10: Tobillo Gyro
    "mag_ank_x", "mag_ank_y", "mag_ank_z",    # 11-13: Tobillo Mag
    "acc_arm_x", "acc_arm_y", "acc_arm_z",    # 14-16: Brazo Acc
    "gyro_arm_x", "gyro_arm_y", "gyro_arm_z", # 17-19: Brazo Gyro
    "mag_arm_x", "mag_arm_y", "mag_arm_z",    # 20-22: Brazo Mag
    "label"                                   # 23: Etiqueta
]

ACTIVITY_LABELS = {
    0: "Null", 1: "Standing", 2: "Sitting", 3: "Lying", 4: "Walking",
    5: "Stairs", 6: "Waist Bend", 7: "Arms Up", 8: "Knees Bend",
    9: "Cycling", 10: "Jogging", 11: "Running", 12: "Jump F&B"
}

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 2. DESCARGA Y EXTRACCIÓN
# ==========================================
def descargar_dataset():
    zip_path = os.path.join(RAW_DIR, "MHEALTHDATASET.zip")
    extracted_folder = os.path.join(RAW_DIR, "MHEALTHDATASET")
    
    if not os.path.exists(extracted_folder):
        print("⬇️ Descargando dataset MHEALTH (puede tardar un poco)...")
        # Header user-agent para evitar bloqueo 403
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        try:
            urllib.request.urlretrieve(DATASET_URL, zip_path)
            print("📦 Descomprimiendo...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            print("✅ Dataset listo.")
        except Exception as e:
            print(f"❌ Error crítico en descarga: {e}")
            print("💡 Solución manual: Descarga el zip de UCI y ponlo en la carpeta 'data_raw'")
            exit()
    else:
        print("✅ Dataset ya existe, saltando descarga.")
    return extracted_folder

# ==========================================
# 3. PREPROCESAMIENTO
# ==========================================
def calcular_magnitud(df, col_x, col_y, col_z):
    return np.sqrt(df[col_x]**2 + df[col_y]**2 + df[col_z]**2)

def cargar_y_procesar_sujetos(data_folder, sujetos_ids):
    data_list = []
    
    for subject in sujetos_ids:
        filename = f"mhealth_subject{subject}.log"
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ No se encontró {filename}")
            continue
            
        print(f"   -> Procesando Sujeto {subject}...")
        try:
            df = pd.read_csv(filepath, sep="\t", header=None)
        except:
            df = pd.read_csv(filepath, delim_whitespace=True, header=None)
            
        df.columns = COLUMNAS_RAW
        
        # Eliminar clase 0
        df = df[df['label'] != 0].copy()
        
        # Ingeniería: Expandir a 26 Canales (Magnitudes)
        df['mag_ch'] = calcular_magnitud(df, 'acc_ch_x', 'acc_ch_y', 'acc_ch_z')
        df['mag_ank'] = calcular_magnitud(df, 'acc_ank_x', 'acc_ank_y', 'acc_ank_z')
        df['mag_arm'] = calcular_magnitud(df, 'acc_arm_x', 'acc_arm_y', 'acc_arm_z')
        
        # Reordenar
        cols_features = [c for c in df.columns if c != 'label']
        df = df[cols_features + ['label']]
        
        data_list.append(df)
        
    if not data_list:
        print("❌ Error: No se cargaron datos. Revisa la carpeta data_raw.")
        exit()

    return pd.concat(data_list, ignore_index=True)

# ==========================================
# 4. VENTANEO Y FEATURES (182 Caracteristicas)
# ==========================================
def extraer_estadisticas(ventana):
    # 26 canales * 7 stats = 182 features
    mean = np.mean(ventana, axis=0)
    std = np.std(ventana, axis=0)
    max_val = np.max(ventana, axis=0)
    min_val = np.min(ventana, axis=0)
    median = np.median(ventana, axis=0)
    ptp = np.ptp(ventana, axis=0)
    var = np.var(ventana, axis=0)
    return np.concatenate([mean, std, max_val, min_val, median, ptp, var])

def generar_dataset_ventaneado(df_raw):
    X_list = []
    y_list = []
    
    datos = df_raw.drop(columns=['label']).values
    etiquetas = df_raw['label'].values
    
    num_samples = datos.shape[0]
    
    print(f"⚙️ Ventaneando {num_samples} filas...")
    
    for start in range(0, num_samples - int(WINDOW_SIZE), STEP_SIZE):
        end = start + int(WINDOW_SIZE)
        
        ventana_data = datos[start:end, :]
        ventana_label = etiquetas[start:end]
        
        # Etiqueta moda
        label_mode = stats.mode(ventana_label, keepdims=True)[0][0]
        
        # Features
        features = extraer_estadisticas(ventana_data)
        
        X_list.append(features)
        y_list.append(label_mode)
        
    return np.array(X_list), np.array(y_list)

# ==========================================
# 5. EJECUCIÓN
# ==========================================
def main():
    base_folder = descargar_dataset()
    
    print("\n--- 1. Carga de Datos ---")
    print("Entrenamiento (Sujetos 1-6)...")
    df_train = cargar_y_procesar_sujetos(base_folder, range(1, 7))
    print("Validación (Sujetos 7-8)...")
    df_val = cargar_y_procesar_sujetos(base_folder, range(7, 9))
    
    print("\n--- 2. Extracción de Features ---")
    X_train, y_train = generar_dataset_ventaneado(df_train)
    X_val, y_val = generar_dataset_ventaneado(df_val)
    
    print(f"Train Shape: {X_train.shape}")
    print(f"Val Shape:   {X_val.shape}")
    
    print("\n--- 3. Estandarización ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Guardar parámetros del scaler
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist()
    }
    with open(os.path.join(OUTPUT_DIR, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f)
    
    print("\n--- 4. Entrenamiento ---")
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    print("\n--- 5. Evaluación ---")
    y_pred = clf.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    print(f"ACCURACY FINAL: {acc:.4f}")
    
    # Matriz de Confusión
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys()) if i!=0],
                yticklabels=[ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys()) if i!=0])
    plt.title(f'Matriz de Confusión\nAccuracy: {acc:.2%}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print("Grafico guardado en 'output/confusion_matrix.png'")
    
    # Exportar Modelo
    joblib.dump(clf, os.path.join(MODELS_DIR, "mhealth_rf_model.pkl"))
    print(f"Modelo guardado en 'models/mhealth_rf_model.pkl'")

if __name__ == "__main__":
    main()