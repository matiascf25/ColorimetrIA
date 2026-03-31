import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import random

def main():
    print("Iniciando Mosaico de Diagnóstico (Hair Extraction)...")
    
    input_csv = "colorimetry_master_index.csv"  # o patched, da igual para las rutas
    output_dir = "debug_hair_boxes"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Limpiar directorio previo si existe
        for f in os.listdir(output_dir):
            if f.endswith(".jpg") or f.endswith(".png"):
                os.remove(os.path.join(output_dir, f))
                
    if not os.path.exists(input_csv):
        print(f"Error: No encuentro el archivo {input_csv}.")
        return

    # 1. Cargar dataset y tomar una muestra aleatoria de 50 imágenes que hayan sido "success"
    print("Cargando rutas de imágenes...")
    df = pd.read_csv(input_csv, low_memory=False)
    
    # Filtramos por éxito para asegurar que hay rostro
    df_valid = df[df['status'] == 'success']
    
    sample_size = min(50, len(df_valid))
    sampled_df = df_valid.sample(n=sample_size, random_state=random.randint(1, 10000))
    image_paths = sampled_df['image_path'].tolist()

    # 2. Inicializar MediaPipe
    base_options = python.BaseOptions(
        model_asset_path='face_landmarker.task',
        delegate=python.BaseOptions.Delegate.CPU # CPU basta para 50 imágenes
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # 3. Procesar y Dibujar
    print(f"Generando mosaicos en la carpeta '{output_dir}/'...")
    for idx, img_path in enumerate(tqdm(image_paths)):
        if not os.path.exists(img_path):
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        results = landmarker.detect(mp_image)
        
        # Copia para dibujar
        annotated_image = image.copy()
        
        color_skin = (0, 0, 255)    # Rojo
        color_hair_L = (0, 255, 0)  # Verde
        color_hair_R = (0, 255, 255)# Amarillo
        
        if results and results.face_landmarks:
            landmarks = results.face_landmarks[0]
            
            # Medidas del rostro para escala
            chin_y = landmarks[152].y * h
            front_y = landmarks[10].y * h
            face_h = chin_y - front_y
            offset_y = int(face_h * 0.12) # 12% más arriba de la frente
            offset_x = int(face_h * 0.1)  # Desplazamiento lateral
            
            # Dibujar Skin ROI (Rojo - Mejilla)
            cx, cy = int(landmarks[234].x * w), int(landmarks[234].y * h)
            cy_c = max(5, min(h-5, cy))
            cx_c = max(5, min(w-5, cx))
            cv2.rectangle(annotated_image, (cx_c-5, cy_c-5), (cx_c+5, cy_c+5), color_skin, 2)
            cv2.putText(annotated_image, "Skin", (cx_c-20, cy_c-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_skin, 1)

            # --- NUEVA LÓGICA DE DETECCIÓN DE CABELLO ---
            # Top Hair (Centro arriba de la frente)
            tx, ty = int(landmarks[10].x * w), int(landmarks[10].y * h) - offset_y
            ty = max(5, min(h-5, ty))
            tx = max(5, min(w-5, tx))
            cv2.rectangle(annotated_image, (tx-5, ty-5), (tx+5, ty+5), (255, 0, 0), 2) # Azul
            cv2.putText(annotated_image, "Top", (tx-15, ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Hair Left (Arriba de la sien izquierda, Landmark 68)
            lx, ly = int(landmarks[68].x * w) - offset_x, int(landmarks[68].y * h) - int(offset_y * 0.7)
            ly = max(5, min(h-5, ly))
            lx = max(5, min(w-5, lx))
            cv2.rectangle(annotated_image, (lx-5, ly-5), (lx+5, ly+5), color_hair_L, 2)
            cv2.putText(annotated_image, "Hair L", (lx-20, ly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_hair_L, 1)
            
            # Hair Right (Arriba de la sien derecha, Landmark 298)
            rx, ry = int(landmarks[298].x * w) + offset_x, int(landmarks[298].y * h) - int(offset_y * 0.7)
            ry = max(5, min(h-5, ry))
            rx = max(5, min(w-5, rx))
            cv2.rectangle(annotated_image, (rx-5, ry-5), (rx+5, ry+5), color_hair_R, 2)
            cv2.putText(annotated_image, "Hair R", (rx-20, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_hair_R, 1)

        # Guardar resultado
        out_name = f"diagnostico_roi_{idx:02d}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), annotated_image)

    print(f"\n✅ ¡Mosaico completado! Revisa la carpeta '{output_dir}' para verificar visualmente que los cuadros caen sobre el cabello y la piel.")

if __name__ == "__main__":
    main()
