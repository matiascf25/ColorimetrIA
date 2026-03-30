import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps, ImageDraw
import os
import math
import numpy as np
import cv2
import mediapipe as mp
from skimage import color
import torch
import torch.nn as nn
import joblib

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="ColorimetrIA Pro: Ojos y Piel", layout="wide")

# 1. DICCIONARIO MAESTRO DE LAS 12 ESTACIONES (HEX)
PALETTES_12 = {
    "Deep Autumn": ["#4E3629", "#5D3A1A", "#3E442D", "#800020", "#B8860B"],
    "Deep Winter": ["#1A1110", "#000080", "#4B0082", "#800000", "#004225"],
    "Light Spring": ["#FFFACD", "#FFB6C1", "#AFEEEE", "#98FB98", "#FFDAB9"],
    "Light Summer": ["#F0F8FF", "#E6E6FA", "#FFC0CB", "#B0E0E6", "#D8BFD8"],
    "True Winter": ["#0000FF", "#FF0000", "#FFFFFF", "#000000", "#FF00FF"],
    "True Summer": ["#4682B4", "#9370DB", "#C71585", "#778899", "#87CEEB"],
    "True Autumn": ["#D2691E", "#8B4513", "#556B2F", "#B22222", "#FF8C00"],
    "True Spring": ["#FF4500", "#32CD32", "#FFD700", "#FF1493", "#00CED1"],
    "Clear Winter": ["#00008B", "#DC143C", "#00FF00", "#F5F5F5", "#191970"],
    "Clear Spring": ["#FF7F50", "#00FF7F", "#FFFF00", "#FF00FF", "#00FFFF"],
    "Soft Summer": ["#B0C4DE", "#BC8F8F", "#708090", "#DDA0DD", "#4682B4"],
    "Soft Autumn": ["#BC8F8F", "#A0522D", "#6B8E23", "#CD853F", "#8FBC8F"]
}

# 2. CARGA DE MODELO PyTorch (ColorNet)
@st.cache_resource
def load_colornet_model():
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    NUM_CLASSES = len(le.classes_)
    
    class ColorNet(nn.Module):
        def __init__(self, num_classes):
            super(ColorNet, self).__init__()
            self.fc1 = nn.Linear(5, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = ColorNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load('biometric_color_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return scaler, le, model

def predict_season_nn(skin_L, skin_b, skin_c, iris_L, iris_b, scaler, le, model):
    # El modelo se entrenó con: ['Skin_L', 'Skin_b', 'Chroma', 'Iris_L', 'Iris_b']
    x = np.array([[skin_L, skin_b, skin_c, iris_L, iris_b]])
    x_scaled = scaler.transform(x)
    
    x_tensor = torch.FloatTensor(x_scaled)
    with torch.no_grad():
        outputs = model(x_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prob, predicted = torch.max(probabilities, 1)
        
    season_name = le.inverse_transform(predicted.numpy())[0]
    confidence = prob.item() * 100
    return season_name, confidence

# 3. MOTOR DE EXTRACCIÓN BIOMÉTRICA (Piel + Iris)
def analyze_face_and_eyes(pil_image):
    image_np = np.array(pil_image)
    model_path = 'face_landmarker.task' 
    
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        detection = landmarker.detect(mp_image)

        if not detection.face_landmarks:
            return None

        h, w, _ = image_np.shape
        
        # Piel: Landmark 234 (Mejilla)
        skin_lm = detection.face_landmarks[0][234]
        sx, sy = int(skin_lm.x * w), int(skin_lm.y * h)
        
        skin_roi = image_np[sy-5:sy+5, sx-5:sx+5]
        s_rgb = np.mean(skin_roi, axis=(0, 1)) / 255.0
        s_lab = color.rgb2lab([[s_rgb]])[0][0]
        
        # Ojo/Iris: Landmark 473 (Pupila Ojo Derecho)
        iris_lm = detection.face_landmarks[0][473]
        ix, iy = int(iris_lm.x * w), int(iris_lm.y * h)
        
        # Extraer patch pequeño del iris (evitar párpados)
        iris_roi = image_np[iy-3:iy+3, ix-3:ix+3]
        i_rgb = np.mean(iris_roi, axis=(0, 1)) / 255.0
        i_lab = color.rgb2lab([[i_rgb]])[0][0]
        
        # Chroma de piel (el iris es menos saturado)
        skin_chroma = math.sqrt(s_lab[1]**2 + s_lab[2]**2)
        
        return {
            "skin_L": s_lab[0], "skin_b": s_lab[2], "skin_c": skin_chroma,
            "iris_L": i_lab[0], "iris_b": i_lab[2],
            "coords": {"skin": (sx, sy), "iris": (ix, iy)}
        }

# --- INTERFAZ ---
try:
    scaler, le, colornet_model = load_colornet_model()
except Exception as e:
    st.error(f"Error cargando modelos: {e}")

st.title("🎨 ColorimetrIA Pro: Inteligencia Facial")

st.header("🎭 Análisis Biométrico y Draping Total")

input_method = st.radio("Elige la fuente de tu fotografía:", ["Sube un Archivo", "Cámara en Vivo"], horizontal=True)

up_file = None
if input_method == "Sube un Archivo":
    up_file = st.file_uploader("Sube tu selfie (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
else:
    up_file = st.camera_input("Tómate una selfie ahora")

if up_file:
    img_orig = Image.open(up_file).convert("RGB")
    
    with st.spinner('Procesando biometría avanzada...'):
        metrics = analyze_face_and_eyes(img_orig)

    if metrics:
        col_img, col_pred = st.columns([1, 1])
        
        with col_img:
            st.image(img_orig, caption="Rostro Detectado", use_container_width=True)
            st.write(f"**LAB Piel:** L: {metrics['skin_L']:.1f} | b: {metrics['skin_b']:.1f} | C: {metrics['skin_c']:.1f}")
            st.write(f"**LAB Iris:** L: {metrics['iris_L']:.1f} | b: {metrics['iris_b']:.1f}")

        with col_pred:
            st.subheader("Resultados de IA (ColorNet)")
            
            # CLASIFICACIÓN MEDIANTE RED NEURONAL Pytorch
            season, confidence = predict_season_nn(metrics['skin_L'], metrics['skin_b'], metrics['skin_c'], metrics['iris_L'], metrics['iris_b'], scaler, le, colornet_model)
            
            st.success(f"### Estación: **{season}**")
            st.caption(f"**Confianza de Predicción:** `{confidence:.2f}%`")
            
            hex_list = PALETTES_12.get(season, ["#CCCCCC"])
            p_cols = st.columns(len(hex_list))
            for i, h in enumerate(hex_list):
                p_cols[i].markdown(f'<div style="background-color:{h}; height:60px; border-radius:8px; border:1px solid white;"></div>', unsafe_allow_html=True)
            
            st.info("¡Estación calculada mediante **ColorNet** (Red Neuronal PyTorch) usando Piel e Iris simultáneamente!")

        # DRAPING TOTAL
        st.divider()
        st.subheader("Simulación de Draping Físico")
        d_cols = st.columns(len(hex_list))
        
        for i, color_m in enumerate(hex_list):
            with d_cols[i]:
                # Aplicamos el marco de color (Draping)
                img_m = ImageOps.expand(img_orig, border=60, fill=color_m)
                st.image(img_m, caption=f"Tonalidad {i+1}", use_container_width=True)
                
    else:
        st.error("Detección facial fallida. Revisa la luz y el ángulo.")
