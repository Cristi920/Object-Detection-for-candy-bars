# ---------------------------------------------------------
# Importăm modulele necesare pentru procesare video, ML și sistem
# ---------------------------------------------------------
import os
import sys
import glob
import time
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------
# Directorul principal al proiectului
# Toate căile sunt definite relativ la acest folder
# ---------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\clisu\Desktop\New folder")

# ---------------------------------------------------------
# Modelul YOLO antrenat
# Verificăm existența fișierului best.pt
# ---------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / r"runs\detect\train2\weights\best.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

# ---------------------------------------------------------
# Folder cu fișiere sample (imagini/video)
# ---------------------------------------------------------
SAMPLES_FOLDER = PROJECT_ROOT / "data/train/images"
if not SAMPLES_FOLDER.exists():
    raise FileNotFoundError(f"❌ Samples folder not found: {SAMPLES_FOLDER}")

# ---------------------------------------------------------
# 🔹 Detectare device (CPU/GPU)
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {DEVICE}")

# ---------------------------------------------------------
# Încărcare bază date nutrițională
# JSON → dicționar optimizat pentru lookup rapid
# ---------------------------------------------------------
NUTRITION_FILE = PROJECT_ROOT / "nutrition_value.json"
with open(NUTRITION_FILE, 'r', encoding='utf-8') as f:
    nutrition_data_raw = json.load(f)
nutrition_data = {k.lower(): v["Declaratie nutritionala / 100g"] for k,v in nutrition_data_raw.items()}

# ---------------------------------------------------------
# Încărcăm modelul YOLO
# ---------------------------------------------------------
model = YOLO(str(MODEL_PATH), task="detect")
labels = model.names

# ---------------------------------------------------------
# Culori pentru bounding boxes
# ---------------------------------------------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# ---------------------------------------------------------
# Variabile pentru FPS tracking
# ---------------------------------------------------------
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# ---------------------------------------------------------
# Nutrienți afișați pe ecran
# Toggle prin taste
# ---------------------------------------------------------
active_display = set([
    'Valoarea energetica', 'Grasimi', 'din care acizi grasi saturati',
    'Glucide', 'din care zaharuri', 'Proteine', 'Sare'
])

key_map = {
    ord('1'): ['Valoarea energetica'],
    ord('2'): ['Grasimi', 'din care acizi grasi saturati'],
    ord('3'): ['Glucide', 'din care zaharuri'],
    ord('4'): ['Proteine'],
    ord('5'): ['Sare']
}

# ---------------------------------------------------------
# Funcții helper pentru parsare valori
# ---------------------------------------------------------
def parse_value(val_str):
    if val_str is None:
        return 0.0
    s = str(val_str).strip()
    s = re.sub(r'[^\d\.,]', '', s)
    s = s.replace(',', '.')
    m = re.search(r'\d+(?:\.\d+)?', s)
    return float(m.group(0)) if m else 0.0

def parse_energy(val_str):
    if val_str is None:
        return 0.0, 0.0
    s = str(val_str).replace(' ', '')
    kj, kcal = 0.0, 0.0
    m_kj = re.search(r'([\d\.,]+)\s*[kK][jJ]', s)
    m_kcal = re.search(r'([\d\.,]+)\s*[kK][cC]a?l?', s)
    if m_kj: kj = parse_value(m_kj.group(1))
    if m_kcal: kcal = parse_value(m_kcal.group(1))
    if kj==0.0 and kcal==0.0:
        nums = re.findall(r'[\d\.,]+', s)
        if len(nums)>=2: kj=parse_value(nums[0]); kcal=parse_value(nums[1])
        elif len(nums)==1: kj=parse_value(nums[0])
    return kj, kcal



# ---------------------------------------------------------
# Funcție desenare bounding box curat
# ---------------------------------------------------------
def format_qty(val, unit, decimals=1):
    if unit=='g': return f"{val:.{decimals}f}".rstrip('0').rstrip('.') + unit
    if unit in ['kJ','kcal']: return str(int(round(val))) + unit
    return f"{val:.{decimals}f}" + unit

def draw_box_clean(img, box, label, color):
    x1, y1, x2, y2 = box
    thickness = 2

    # bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # label background
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )

    y1_label = max(y1 - th - 8, 0)

    cv2.rectangle(
        img,
        (x1, y1_label),
        (x1 + tw + 6, y1),
        color,
        -1
    )

    # label text (alb, clar)
    cv2.putText(
        img,
        label,
        (x1 + 3, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

# ---------------------------------------------------------
# Alegere mod (imagine / video / webcam)
# ---------------------------------------------------------
print("1: Video")
print("2: Imagine")  
print("3: Webcam")  
print("4: Cancel","\n")
mod_ales = int(input("Alege modul: "))

if mod_ales==1:
    files = [f for f in os.listdir(SAMPLES_FOLDER) if f.lower().endswith((".mp4",".avi",".mov"))]
    if not files: raise ValueError("❌ Nu am găsit video în folder.")
    source_type='video'
    cap = cv2.VideoCapture(str(SAMPLES_FOLDER/files[0]))
elif mod_ales==2:
    files = [f for f in os.listdir(SAMPLES_FOLDER) if f.lower().endswith((".jpg",".png",".jpeg"))]
    if not files: raise ValueError("❌ Nu am găsit imagini în folder.")
    source_type='image'
    imgs_list = [str(SAMPLES_FOLDER/files[0])]
elif mod_ales==3:
    source_type='usb'
    cap = cv2.VideoCapture(0)
else:
    sys.exit("!! Canceled !!")

# ---------------------------------------------------------
# Selectare rezoluție imagine
# ---------------------------------------------------------
print("Select image size:")
print("1: 480x480")
print("2: 640x480")
print("3: 1280x720")
size_option = int(input("Alege dimensiunea: "))
if size_option == 1:
    IMG_SIZE = (480,480)
elif size_option == 2:
    IMG_SIZE = (640,480)
elif size_option == 3:
    IMG_SIZE = (1280,720)
else:
    IMG_SIZE = (640,480)

# ---------------------------------------------------------
# Loop principal de procesare
# ---------------------------------------------------------
prev_object_count = -1  # pentru a afișa doar la schimbări

while True:
    t_start = time.perf_counter()

    # load frame
    if source_type=='image':
        if img_count>=len(imgs_list):
            print("All images processed. Exiting.")
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count+=1
    elif source_type in ['video','usb']:
        ret, frame = cap.read()
        if not ret or frame is None: break

    frame_to_display = cv2.resize(frame, IMG_SIZE)
# ---------------------------------------------------------
# Inițializare totaluri nutriționale
# ---------------------------------------------------------
    totals_numeric = {
        'Valoarea energetica_kj': 0.0,
        'Valoarea energetica_kcal': 0.0,
        'Grasimi': 0.0,
        'din care acizi grasi saturati': 0.0,
        'Glucide': 0.0,
        'din care zaharuri': 0.0,
        'Proteine': 0.0,
        'Sare': 0.0
    }
    detected_products = []
# ---------------------------------------------------------
# Rulare YOLO pe frame
# ---------------------------------------------------------
    results = model(frame_to_display, verbose=False)
    detections = results[0].boxes
    object_count=0

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx=int(det.cls.item())
        classname=labels[classidx]
        conf=det.conf.item()

        if conf>=0.5:
            color = bbox_colors[classidx % 10]
            label = f"{classname} {int(conf*100)}%"
            draw_box_clean(
                frame_to_display,
                (xmin, ymin, xmax, ymax),
                label,
                color
            )
            object_count+=1

            # ---------------------------------------------------------
            # Actualizare valori nutriționale
            # ---------------------------------------------------------

            cname_lower = classname.lower()
            if cname_lower in nutrition_data:
                info = {k.lower():v for k,v in nutrition_data[cname_lower].items()}
                kj_add,kcal_add=parse_energy(info.get('valoarea energetica') or info.get('valoarea energetică'))
                totals_numeric['Valoarea energetica_kj']+=kj_add
                totals_numeric['Valoarea energetica_kcal']+=kcal_add
                for key in ['Grasimi','din care acizi grasi saturati','Glucide','din care zaharuri','Proteine','Sare']:
                    val = parse_value(info.get(key.lower()))
                    totals_numeric[key]+=val
                if classname not in detected_products:
                    detected_products.append(classname)
    # ---------------------------------------------------------
    # Afișare info în terminal
    # ---------------------------------------------------------
    if object_count != prev_object_count:
        print(f"Objects detected: {object_count}")
        prev_object_count = object_count

    # ---------------------------------------------------------
    # Formatare valori pentru display
    # ---------------------------------------------------------
    energy_str=f"{format_qty(totals_numeric['Valoarea energetica_kj'],'kJ')}/{format_qty(totals_numeric['Valoarea energetica_kcal'],'kcal')}"
    display_totals = {
        'Valoarea energetica': energy_str,
        'Grasimi': format_qty(totals_numeric['Grasimi'],'g'),
        'din care acizi grasi saturati': format_qty(totals_numeric['din care acizi grasi saturati'],'g'),
        'Glucide': format_qty(totals_numeric['Glucide'],'g'),
        'din care zaharuri': format_qty(totals_numeric['din care zaharuri'],'g'),
        'Proteine': format_qty(totals_numeric['Proteine'],'g'),
        'Sare': format_qty(totals_numeric['Sare'],'g')
    }
    # ---------------------------------------------------------
    # Draw totals
    # ---------------------------------------------------------
    line_height = 25
    padding_bottom = 20

    y_start = frame_to_display.shape[0] - padding_bottom - line_height * len(active_display)
    for idx,key in enumerate(display_totals):
        if key in active_display:
            cv2.putText(frame_to_display,f"{key}: {display_totals[key]}",(10,y_start+idx*25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    # ---------------------------------------------------------
    # Afișare obiecte detectate
    # ---------------------------------------------------------
    cv2.putText(frame_to_display,f'Objects: {object_count}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.imshow("YOLO Detection", frame_to_display)


    # ---------------------------------------------------------
    # Control tastatură
    # ---------------------------------------------------------

    key=cv2.waitKey(0 if source_type=='image' else 5)
    if key in key_map:
        for nutrient in key_map[key]:
            if nutrient in active_display: active_display.remove(nutrient)
            else: active_display.add(nutrient)
    elif key in [ord('q'),ord('Q')]:
        break
    elif key in [ord('p'),ord('P')]:
        cv2.imwrite('capture.png', frame)

    
    # ---------------------------------------------------------
    # Calcul FPS mediu
    # ---------------------------------------------------------
    t_stop=time.perf_counter()
    frame_rate_calc=1/(t_stop-t_start) if (t_stop-t_start)>0 else 0.0
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer)>fps_avg_len: frame_rate_buffer.pop(0)
    avg_frame_rate=np.mean(frame_rate_buffer)


# ---------------------------------------------------------
# Cleanup final
# ---------------------------------------------------------
print(f"Average FPS: {avg_frame_rate:.2f}")
if source_type in ['video','usb']: cap.release()
cv2.destroyAllWindows()
