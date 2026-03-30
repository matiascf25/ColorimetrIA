import pandas as pd
import mediapipe as mp
import numpy as np
import cv2
import math
from skimage import color
from tqdm import tqdm

# --- CONFIGURACIÓN TÉCNICA ---
def get_12_seasons_logic(L, a, b, chroma):
    if L < 35: return "Deep Autumn" if b > 0 else "Deep Winter"
    if L > 70: return "Light Spring" if b > 0 else "Light Summer"
    if chroma > 28: return "Clear Spring" if b > 0 else "Clear Winter"
    if chroma < 14: return "Soft Autumn" if b > 0 else "Soft Summer"
    if b > 12: return "True Autumn" if L < 55 else "True Spring"
    if b < -12: return "True Winter" if L < 55 else "True Summer"
    return "Neutral / Undefined"

# MediaPipe con aceleración por GPU (RTX 5060 Ti)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task', delegate=BaseOptions.Delegate.GPU),
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)

# --- PROCESAMIENTO ---
df = pd.read_csv('colorimetry_master_index.csv')
df = df[df['status'] == 'success'].copy()
print(f"Matias, iniciando el parcheo de {len(df):,} registros...")

iris_L_list, iris_b_list, season_12_list = [], [], []

with FaceLandmarker.create_from_options(options) as landmarker:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 1. Cargar imagen y detectar iris
            img = cv2.imread(row['image_path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection = landmarker.detect(mp_image)
            
            if detection.face_landmarks:
                h, w, _ = img_rgb.shape
                # Landmark 473 (Pupila derecha)
                lm = detection.face_landmarks[0][473]
                ix, iy = int(lm.x * w), int(lm.y * h)
                roi = img_rgb[max(0, iy-3):min(h, iy+3), max(0, ix-3):min(w, ix+3)]
                avg_rgb = np.mean(roi, axis=(0, 1)) / 255.0
                i_lab = color.rgb2lab([[avg_rgb]])[0][0]
                
                i_L, i_b = i_lab[0], i_lab[2]
            else:
                i_L, i_b = np.nan, np.nan
            
            # 2. Aplicar lógica de 12 estaciones
            chroma = math.sqrt(row['Skin_a']**2 + row['Skin_b']**2)
            season = get_12_seasons_logic(row['Skin_L'], row['Skin_a'], row['Skin_b'], chroma)
            
            iris_L_list.append(i_L)
            iris_b_list.append(i_b)
            season_12_list.append(season)
            
        except:
            iris_L_list.append(np.nan)
            iris_b_list.append(np.nan)
            season_12_list.append("Error")

# Guardar el nuevo dataset "Gold"
df['Iris_L'] = iris_L_list
df['Iris_b'] = iris_b_list
df['Season_12'] = season_12_list
df.dropna(subset=['Iris_L', 'Season_12'], inplace=True)
df.to_csv('colorimetry_gold_standard.csv', index=False)
print("✅ Dataset 'Gold Standard' listo para el entrenamiento Pro.")
