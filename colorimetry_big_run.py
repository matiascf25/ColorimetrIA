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
global_celeba_df = None

def init_worker():
    global global_landmarker
    global global_celeba_df
    
    celeba_csv_path = "Dataset de Atributos (Cabello y Ojos)/CelebFacesA/list_attr_celeba.csv"
    if os.path.exists(celeba_csv_path):
        global_celeba_df = pd.read_csv(celeba_csv_path)

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

def get_refined_season(L, a, b, contrast):
    if L is None or a is None or b is None or contrast is None:
        return "Unknown"
        
    chroma = math.sqrt(a**2 + b**2)
    temp = "Warm" if b > 0 else "Cold"
    
    if temp == "Cold":
        if contrast == "High" and chroma > 20:
            return "Clear Winter"
        if contrast == "Low" and chroma < 15:
            return "Soft Summer"
        return "Winter/Summer General"
        
    if temp == "Warm":
        if contrast == "High" and chroma > 20:
            return "Clear Spring"
        if contrast == "Low" and chroma < 15:
            return "Soft Autumn"
        return "Autumn/Spring General"

def get_celeba_hair_l_fallback(image_name, celeba_df):
    if celeba_df is None or image_name not in celeba_df['image_id'].values:
        return None
    
    row = celeba_df[celeba_df['image_id'] == image_name].iloc[0]
    
    if row['Bald'] == 1:
        return 'Bald'
    elif row['Black_Hair'] == 1:
        return 15.0
    elif row['Brown_Hair'] == 1:
        return 35.0
    elif row['Gray_Hair'] == 1:
        return 60.0
    elif row['Blond_Hair'] == 1:
        return 75.0
    return None

def process_single_image(img_path_str):
    global global_landmarker
    global global_celeba_df
    
    img_path = Path(img_path_str)
    img_name = img_path.name
    
    status = "failed"
    metrics = None
    
    try:
        image = cv2.imread(img_path_str)
        if image is not None:
            h, w, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            results = global_landmarker.detect(mp_image)
            
            if results and results.face_landmarks:
                landmarks = results.face_landmarks[0]
                
                cx, cy = int(landmarks[234].x * w), int(landmarks[234].y * h)
                cy = max(5, min(h-5, cy))
                cx = max(5, min(w-5, cx))
                
                roi_skin = rgb_image[cy-5:cy+5, cx-5:cx+5]
                if roi_skin.size > 0:
                     avg_rgb_skin = np.mean(roi_skin, axis=(0, 1)) / 255.0
                     lab_skin = color.rgb2lab([[avg_rgb_skin]])[0][0]
                     skin_L, skin_a, skin_b = lab_skin[0], lab_skin[1], lab_skin[2]
                     
                     hair_L_extracted = None
                     lx, ly = int(landmarks[162].x * w), int(landmarks[162].y * h)
                     hx_l = max(0, lx - int(w * 0.05))
                     try_hair_l = rgb_image[max(0, ly-5):min(h, ly+5), max(0, hx_l-5):min(w, hx_l+5)]
                     
                     if try_hair_l.size > 0:
                         avg_rgb_hair = np.mean(try_hair_l, axis=(0, 1)) / 255.0
                         hair_L_extracted = color.rgb2lab([[avg_rgb_hair]])[0][0][0]
                     else:
                         rx, ry = int(landmarks[389].x * w), int(landmarks[389].y * h)
                         hx_r = min(w, rx + int(w * 0.05))
                         try_hair_r = rgb_image[max(0, ry-5):min(h, ry+5), max(0, hx_r-5):min(w, hx_r+5)]
                         if try_hair_r.size > 0:
                             avg_rgb_hair = np.mean(try_hair_r, axis=(0, 1)) / 255.0
                             hair_L_extracted = color.rgb2lab([[avg_rgb_hair]])[0][0][0]

                     metrics = {
                         "Skin_L": skin_L,
                         "Skin_a": skin_a,
                         "Skin_b": skin_b,
                         "Hair_L_extracted": hair_L_extracted
                     }
                     status = "success"
                else:
                     status = "discarded_roi_skin"
            else:
                status = "discarded"
        else:
            status = "discarded_reading"
    except Exception as e:
        status = "error"

    row_data = {
        "image_path": img_path_str,
        "status": status,
        "Skin_L": None,
        "Skin_a": None,
        "Skin_b": None,
        "Chroma": None,
        "Hair_L": None,
        "Hair_Source": None,
        "Temperature": None,
        "Contrast_Type": None,
        "Season_Hint": None
    }
    
    if metrics:
        row_data["Skin_L"] = metrics["Skin_L"]
        row_data["Skin_a"] = metrics["Skin_a"]
        row_data["Skin_b"] = metrics["Skin_b"]
        row_data["Chroma"] = math.sqrt(row_data["Skin_a"]**2 + row_data["Skin_b"]**2)
        
        row_data["Temperature"] = "Warm" if row_data["Skin_b"] > 0 else "Cold"

        hair_l = metrics["Hair_L_extracted"]
        hair_source = "MediaPipe_Temple"
        
        if hair_l is None:
            celeba_fallback = get_celeba_hair_l_fallback(img_name, global_celeba_df)
            if celeba_fallback == 'Bald':
                hair_l = row_data["Skin_L"]
                row_data["Contrast_Type"] = "Low"
                hair_source = "CelebA_Bald"
            elif celeba_fallback is not None:
                hair_l = celeba_fallback
                hair_source = "CelebA_Fallback"
        
        row_data["Hair_L"] = hair_l
        row_data["Hair_Source"] = hair_source if hair_l is not None else "Unknown"
        
        if row_data["Contrast_Type"] is None and hair_l is not None and row_data["Skin_L"] is not None:
            diff = abs(row_data["Skin_L"] - float(hair_l))
            if diff > 40:
                row_data["Contrast_Type"] = "High"
            elif diff > 20:
                row_data["Contrast_Type"] = "Medium"
            else:
                row_data["Contrast_Type"] = "Low"

        row_data["Season_Hint"] = get_refined_season(
            row_data["Skin_L"], row_data["Skin_a"], row_data["Skin_b"], row_data["Contrast_Type"]
        )
        
    return row_data

def main():
    print("Iniciando Pipeline de Alto Rendimiento...")
    datasets_dirs = [
        "Dataset Segmentación",
        "Dataset Tono de Piel (Core de Colorimetría)",
        "Dataset de Atributos (Cabello y Ojos)"
    ]
    
    output_csv = "colorimetry_master_index.csv"
    processed_files = set()
    
    # 1. Sistema de Resiliencia: Cargar archivos ya procesados
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv, usecols=["image_path"], low_memory=False)
        processed_files = set(df_existing["image_path"].values)
        print(f"Detectado archivo previo. {len(processed_files)} imágenes ya fueron procesadas y serán omitidas.")
    
    # 2. Búsqueda Rápida
    print("Recolectando imágenes globales. Por favor, espera...")
    all_images = []
    for d in datasets_dirs:
        for p in Path(d).rglob("*.jpg"):
            path_str = str(p)
            if path_str not in processed_files:
                all_images.append(path_str)
            
    total_imgs = len(all_images)
    print(f"Total de nuevas imágenes en cola para clasificar: {total_imgs}")
    
    if total_imgs == 0:
        print("Todas las imágenes han sido procesadas o no se encontraron imágenes.")
        return

    # 3. Paralelización y Batch Saving
    BATCH_SIZE = 500
    batch_results = []
    
    # max_workers=12 de los 16 posibles
    workers = min(12, total_imgs, os.cpu_count() or 12)
    print(f"Lanzando un ProcessPool con {workers} hilos...")
    print("BaseOptions.Delegate.GPU habilitado de preferencia...")

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        # Enviar todas las tareas (futures)
        futures = {executor.submit(process_single_image, img): img for img in all_images}
        
        # Barra de progreso tqdm combinada con as_completed
        with tqdm(total=total_imgs, desc="Procesando rostros") as pbar:
            for future in as_completed(futures):
                try:
                    data = future.result()
                    batch_results.append(data)
                except Exception as exc:
                    img_path_failed = futures[future]
                    batch_results.append({
                        "image_path": img_path_failed,
                        "status": f"exception_{str(exc)}"
                    })
                
                # Batch Saving trigger
                if len(batch_results) >= BATCH_SIZE:
                    df_batch = pd.DataFrame(batch_results)
                    write_header = not os.path.exists(output_csv)
                    df_batch.to_csv(output_csv, mode='a', header=write_header, index=False)
                    batch_results.clear()
                
                pbar.update(1)

            # Flush cualquier residuo en el batch list a disco
            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                write_header = not os.path.exists(output_csv)
                df_batch.to_csv(output_csv, mode='a', header=write_header, index=False)
                batch_results.clear()
                
    print(f"Proceso maestro finalizado. Logs appendados en {output_csv}.")

if __name__ == "__main__":
    main()
