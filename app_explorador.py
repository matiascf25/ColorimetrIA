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

# DICCIONARIO DE COLORES A EVITAR (HEX)
PALETTES_AVOID_12 = {
    "Deep Autumn": ["#F0F8FF", "#FFE4E1", "#E6E6FA", "#00FFFF", "#FF00FF"],
    "Deep Winter": ["#FFA500", "#FFD700", "#D2691E", "#FF8C00", "#BDB76B"],
    "Light Spring": ["#000000", "#191970", "#4B0082", "#2F4F4F", "#8B0000"],
    "Light Summer": ["#8B4513", "#D2691E", "#FF4500", "#556B2F", "#B8860B"],
    "True Winter": ["#DAA520", "#CD853F", "#8B4513", "#D2691E", "#556B2F"],
    "True Summer": ["#FF8C00", "#FFD700", "#FF4500", "#8B4513", "#B8860B"],
    "True Autumn": ["#FF00FF", "#00FFFF", "#FF1493", "#0000FF", "#E6E6FA"],
    "True Spring": ["#708090", "#2F4F4F", "#778899", "#A9A9A9", "#000000"],
    "Clear Winter": ["#F5DEB3", "#D2B48C", "#808000", "#CD853F", "#A0522D"],
    "Clear Spring": ["#4682B4", "#778899", "#808080", "#708090", "#C0C0C0"],
    "Soft Summer": ["#FF0000", "#FF4500", "#FFFF00", "#000000", "#00FF00"],
    "Soft Autumn": ["#0000FF", "#FF00FF", "#FFFFFF", "#00FFFF", "#DC143C"]
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
            self.fc1 = nn.Linear(7, 128)
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

def predict_season_nn(skin_L, skin_b, skin_c, iris_L, iris_b, hair_L, hair_b, scaler, le, model):
    # El modelo se entrenó con: ['Skin_L', 'Skin_b', 'Chroma', 'Iris_L', 'Iris_b', 'Hair_L', 'Hair_b']
    x = np.array([[skin_L, skin_b, skin_c, iris_L, iris_b, hair_L, hair_b]])
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
        
        # Medidas del rostro para escala
        chin_y = detection.face_landmarks[0][152].y * h
        front_y = detection.face_landmarks[0][10].y * h
        face_h = chin_y - front_y
        offset_y = int(face_h * 0.12) # 12% más arriba de la frente
        offset_x = int(face_h * 0.1)  # Desplazamiento lateral

        # Cabello: Sien izquierda alta (68) o derecha alta (298)
        landmarks = detection.face_landmarks[0]
        hair_L = None
        hair_b = None
        
        # Hair Left
        lx, ly = int(landmarks[68].x * w) - offset_x, int(landmarks[68].y * h) - int(offset_y * 0.7)
        ly = max(5, min(h-5, ly))
        lx = max(5, min(w-5, lx))
        roi_hair = image_np[ly-5:ly+5, lx-5:lx+5]
        
        if roi_hair.size > 0:
            h_rgb = np.mean(roi_hair, axis=(0, 1)) / 255.0
            h_lab = color.rgb2lab([[h_rgb]])[0][0]
            hair_L = h_lab[0]
            hair_b = h_lab[2]
        else:
            # Hair Right
            rx, ry = int(landmarks[298].x * w) + offset_x, int(landmarks[298].y * h) - int(offset_y * 0.7)
            ry = max(5, min(h-5, ry))
            rx = max(5, min(w-5, rx))
            roi_hair_r = image_np[ry-5:ry+5, rx-5:rx+5]
            if roi_hair_r.size > 0:
                h_rgb = np.mean(roi_hair_r, axis=(0, 1)) / 255.0
                h_lab = color.rgb2lab([[h_rgb]])[0][0]
                hair_L = h_lab[0]
                hair_b = h_lab[2]
                
        # Fallback de emergencia si no detecta cabello
        if hair_L is None:
            hair_L = s_lab[0]
            hair_b = s_lab[2]
            h_rgb = s_rgb
        
        # Chroma de piel (el iris es menos saturado)
        skin_chroma = math.sqrt(s_lab[1]**2 + s_lab[2]**2)
        
        return {
            "skin_L": s_lab[0], "skin_b": s_lab[2], "skin_c": skin_chroma,
            "iris_L": i_lab[0], "iris_b": i_lab[2],
            "hair_L": hair_L, "hair_b": hair_b,
            "coords": {"skin": (sx, sy), "iris": (ix, iy)},
            "rgb": {
                "skin": f"rgb({int(s_rgb[0]*255)}, {int(s_rgb[1]*255)}, {int(s_rgb[2]*255)})",
                "iris": f"rgb({int(i_rgb[0]*255)}, {int(i_rgb[1]*255)}, {int(i_rgb[2]*255)})",
                "hair": f"rgb({int(h_rgb[0]*255)}, {int(h_rgb[1]*255)}, {int(h_rgb[2]*255)})" if 'h_rgb' in locals() else f"rgb({int(s_rgb[0]*255)}, {int(s_rgb[1]*255)}, {int(s_rgb[2]*255)})"
            }
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
            
            # Swatches de lo que ve el modelo
            st.markdown("#### Lo que la IA extrajo para su decisión:")
            swatch_html = f"""
            <div style="display: flex; gap: 15px; margin-bottom: 15px;">
                <div style="text-align: center;">
                    <div style="width: 60px; height: 60px; border-radius: 8px; background-color: {metrics['rgb']['skin']}; border: 1px solid #888; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);"></div>
                    <small><b>Piel</b></small>
                </div>
                <div style="text-align: center;">
                    <div style="width: 60px; height: 60px; border-radius: 8px; background-color: {metrics['rgb']['iris']}; border: 1px solid #888; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);"></div>
                    <small><b>Iris</b></small>
                </div>
                <div style="text-align: center;">
                    <div style="width: 60px; height: 60px; border-radius: 8px; background-color: {metrics['rgb']['hair']}; border: 1px solid #888; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);"></div>
                    <small><b>Cabello</b></small>
                </div>
            </div>
            """
            st.markdown(swatch_html, unsafe_allow_html=True)
            
            st.write(f"**LAB Piel:** L: {metrics['skin_L']:.1f} | b: {metrics['skin_b']:.1f} | C: {metrics['skin_c']:.1f}")
            st.write(f"**LAB Iris:** L: {metrics['iris_L']:.1f} | b: {metrics['iris_b']:.1f}")
            st.write(f"**LAB Cabello:** L: {metrics['hair_L']:.1f} | b: {metrics['hair_b']:.1f}")

        with col_pred:
            st.subheader("Resultados de IA (ColorNet)")
            
            # CLASIFICACIÓN MEDIANTE RED NEURONAL Pytorch
            season, confidence = predict_season_nn(metrics['skin_L'], metrics['skin_b'], metrics['skin_c'], metrics['iris_L'], metrics['iris_b'], metrics['hair_L'], metrics['hair_b'], scaler, le, colornet_model)
            
            st.success(f"### Estación: **{season}**")
            st.caption(f"**Confianza de Predicción:** `{confidence:.2f}%`")
            
            hex_list = PALETTES_12.get(season, ["#CCCCCC"])
            p_cols = st.columns(len(hex_list))
            for i, h in enumerate(hex_list):
                p_cols[i].markdown(f'<div style="background-color:{h}; height:40px; border-radius:8px; border:1px solid white;"></div>', unsafe_allow_html=True)
            
            st.markdown("##### ❌ Colores a Evitar")
            hex_avoid = PALETTES_AVOID_12.get(season, ["#000000"])
            a_cols = st.columns(len(hex_avoid))
            for i, h in enumerate(hex_avoid):
                a_cols[i].markdown(f'<div style="background-color:{h}; height:40px; border-radius:8px; border:1px solid #333; opacity:0.8;"></div>', unsafe_allow_html=True)
                
            st.info("¡Estación calculada mediante **ColorNet** (Red Neuronal PyTorch) usando Piel, Iris y Cabello simultáneamente!")

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
