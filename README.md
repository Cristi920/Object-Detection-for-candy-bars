YOLO Candy Bar Detection & Nutrition Values
==================================================

Descriere
---------
Acest proiect implementează un pipeline complet pentru antrenarea unui model YOLO
custom și pentru detecția în timp real a produselor alimentare cu calcul automat
al valorilor nutriționale.

Include:
- pregătirea dataset-ului
- split train/validation
- generare automată data.yaml
- training YOLO
- testare model
- detecție în timp real
- agregare nutrițională live

Tehnologii:
Python, Ultralytics YOLO, PyTorch, OpenCV,label-studio

Pentru a genera datasetul necesar,data.zip, putem apela la label-studio care oferă ușurință în a crea label-urile necesare pentru imaginile care le avem , 
pentru al rula în Visual-Code rulăm comanda următoare în Terminal, label-studio

Structura proiectului
==================================================

project/
│
├── prepare_data.py
├── train_val_split.py
├── create_data_yaml.py
├── train.py
├── test_model.py
├── yolo_detect.py
│
├── data.zip
├── custom_data/
├── data/
│   ├── train/
│   ├── validation/
│   └── data.yaml
│
├── classes.txt
├── nutrition_value.json
└── README.txt


Cerințe
==================================================

Python 3.12.3
ultralytics
opencv-python
pyyaml
numpy
PyTorch

Instalare:

pip install -r
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Ordinea de rulare
==================================================

1. python prepare_data.py
2. python create_data_yaml.py
3. python train.py
4. python test_model.py
5. python yolo_detect.py


prepare_data.py
==================================================

- extrage dataset-ul din data.zip
- creează structura YOLO
- rulează split automat train/validation


train_val_split.py
==================================================

- împarte dataset-ul în:
  90% train
  10% validation

Rulare manuală:

python train_val_split.py --datapath custom_data --train_pct 0.9


create_data_yaml.py
==================================================

- citește clasele din classes.txt
- generează fișierul data.yaml
- creează config pentru YOLO


train.py
==================================================

- detectează automat GPU/CPU
- antrenează modelul YOLO
- salvează rezultatele în runs/detect/

Model implicit:
yolo11l.pt

Modele alternative:
yolo11n (rapid)
yolo11s (light)
yolo11m (echilibrat)
yolo11l (precizie mare)
yolo11x (maximă precizie)


test_model.py
==================================================

- rulează inferență pe imaginile de validation
- salvează rezultate în runs/detect/predict/


yolo_detect.py
==================================================

Aplicație de detecție în timp real.

Moduri:
1 = video
2 = imagine
3 = webcam

Funcții:
- bounding boxes curate
- overlay nutrițional live
- FPS tracking
- captură imagine
- toggle nutrienți


Controale tastatură
==================================================

1 → energie
2 → grăsimi
3 → glucide
4 → proteine
5 → sare
P → screenshot
Q → ieșire


Note importante
==================================================

- dataset-ul trebuie să fie în format YOLO
- fiecare imagine trebuie să aibă label .txt
- classes.txt trebuie să existe
- numele claselor trebuie să corespundă cu nutrition_value.json


Utilizări
==================================================

- recunoaștere produse alimentare
- analiză nutrițională automată
- proiecte AI / computer vision
- cercetare
- prototip aplicații smart diet
