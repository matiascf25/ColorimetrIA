import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from skimage import color

def extract_color_metrics(image_path):
    # Inicializar MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        image = cv2.imread(image_path)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None

        # Coordenadas simplificadas para ROI (Mejilla, Iris, Pelo)
        # Aquí Gemini puede ayudarte a expandir los landmarks exactos
        h, w, _ = image.shape
        # Ejemplo: Punto de la mejilla (Landmark 234)
        cx, cy = int(results.multi_face_landmarks[0].landmark[234].x * w), \
                 int(results.multi_face_landmarks[0].landmark[234].y * h)
        
        roi_skin = image[cy-5:cy+5, cx-5:cx+5]
        avg_rgb = np.mean(roi_skin, axis=(0, 1)) / 255.0
        
        # Conversión a CIELAB
        lab = color.rgb2lab([[avg_rgb]])[0][0]
        
        return {
            "L": lab[0], "a": lab[1], "b": lab[2],
            "path": image_path
        }