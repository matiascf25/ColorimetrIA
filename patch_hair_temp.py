import os
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from skimage import color
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Variables globales para cada worker
global_landmarker = None

def init_worker():
    global global_landmarker
    
    try:
        base_options = python.BaseOptions(
            model_asset_path='face_landmarker.task',
            delegate=python.BaseOptions.Delegate.GPU
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        global_landmarker = vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        # Fallback to CPU if GPU delegate fails
        base_options = python.BaseOptions(
            model_asset_path='face_landmarker.task',
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        global_landmarker = vision.FaceLandmarker.create_from_options(options)

def process_single_image(row_data):
    global global_landmarker
    
    img_path_str = row_data['image_path']
    skin_b = row_data.get('Skin_b', 0.0) # Fallback if needed
    
    hair_b = None
    
    try:
        image = cv2.imread(img_path_str)
        if image is not None:
            h, w, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            results = global_landmarker.detect(mp_image)
            
            if results and results.face_landmarks:
                landmarks = results.face_landmarks[0]
                
                # Medidas del rostro para escala
                chin_y = landmarks[152].y * h
                front_y = landmarks[10].y * h
                face_h = chin_y - front_y
                offset_y = int(face_h * 0.12) # 12% más arriba de la frente
                offset_x = int(face_h * 0.1)  # Desplazamiento lateral
                
                # Hair Left (Arriba de la sien izquierda, Landmark 68)
                lx, ly = int(landmarks[68].x * w) - offset_x, int(landmarks[68].y * h) - int(offset_y * 0.7)
                ly = max(5, min(h-5, ly))
                lx = max(5, min(w-5, lx))
                
                try_hair_l = rgb_image[ly-5:ly+5, lx-5:lx+5]
                
                if try_hair_l.size > 0:
                    avg_rgb_hair = np.mean(try_hair_l, axis=(0, 1)) / 255.0
                    hair_b = color.rgb2lab([[avg_rgb_hair]])[0][0][2]
                else:
                    # Hair Right (Arriba de la sien derecha, Landmark 298)
                    rx, ry = int(landmarks[298].x * w) + offset_x, int(landmarks[298].y * h) - int(offset_y * 0.7)
                    ry = max(5, min(h-5, ry))
                    rx = max(5, min(w-5, rx))
                    
                    try_hair_r = rgb_image[ry-5:ry+5, rx-5:rx+5]
                    if try_hair_r.size > 0:
                        avg_rgb_hair = np.mean(try_hair_r, axis=(0, 1)) / 255.0
                        hair_b = color.rgb2lab([[avg_rgb_hair]])[0][0][2]
    except Exception as e:
        pass
        
    # Fallback to skin_b if extracting hair fails
    if pd.isna(hair_b) or hair_b is None:
        hair_b = skin_b

    row_data['Hair_b'] = hair_b
    return row_data

def main():
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    print("Iniciando Patcheo de Hair_b (Temperatura del Cabello)...")
    
    input_csv = "colorimetry_master_index.csv"
    output_csv = "colorimetry_master_index_patched.csv"
    
    if not os.path.exists(input_csv):
        print(f"File {input_csv} no encontrado.")
        return
        
    df = pd.read_csv(input_csv, low_memory=False)
    
    # Check if we already patched
    if 'Hair_b' in df.columns:
        # maybe we want to re-run only on those without Hair_b? No, let's just assert if they want full run
        pass
        
    # Let's convert df to a list of dicts for processing
    rows_to_process = df.to_dict('records')
    total_imgs = len(rows_to_process)
    
    print(f"Total de imágenes a parchear: {total_imgs}")
    
    BATCH_SIZE = 500
    batch_results = []
    
    workers = min(12, total_imgs, os.cpu_count() or 12)
    print(f"Lanzando un ProcessPool con {workers} hilos...")

    # We will write to a new CSV file
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        with tqdm(total=total_imgs, desc="Parcheando Hair_b") as pbar:
            # Map chunkeará el envío de parámetros, minimizando el impacto en RAM
            for data in executor.map(process_single_image, rows_to_process, chunksize=50):
                batch_results.append(data)
                
                if len(batch_results) >= BATCH_SIZE:
                    df_batch = pd.DataFrame(batch_results)
                    write_header = not os.path.exists(output_csv)
                    df_batch.to_csv(output_csv, mode='a', header=write_header, index=False)
                    batch_results.clear()
                
                pbar.update(1)

            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                write_header = not os.path.exists(output_csv)
                df_batch.to_csv(output_csv, mode='a', header=write_header, index=False)
                batch_results.clear()
                
    print(f"Patcheo finalizado. Nuevo archivo guardado como {output_csv}.")

if __name__ == "__main__":
    main()
