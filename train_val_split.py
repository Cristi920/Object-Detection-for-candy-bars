# ---------------------------------------------------------
# Importăm moduele necesare
# ---------------------------------------------------------
from pathlib import Path
import random
import os
import sys
import shutil
import argparse


# ---------------------------------------------------------
# CONFIG PROIECT
# Directorul principal unde se vor salva folderele train/validation
# ---------------------------------------------------------
PROJECT_ROOT = Path(
    r"C:\Users\clisu\Desktop\New folder"
)


# ---------------------------------------------------------
# ARGUMENTE DIN LINIA DE COMANDĂ
# Permite rularea scriptului cu parametri dinamici
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', required=True,
                    help='Folder care conține images/ și labels/')
parser.add_argument('--train_pct', default=0.8, type=float,
                    help='Procent train (ex: 0.9)')
args = parser.parse_args()

data_path = Path(args.datapath)
train_percent = args.train_pct


# ---------------------------------------------------------
# VALIDĂRI INPUT
# Verificăm dacă parametrii sunt corecți înainte de rulare
# ---------------------------------------------------------
if not data_path.is_dir():
    print('❌ datapath invalid')
    sys.exit(1)

if not (0.01 < train_percent < 0.99):
    print('❌ train_pct trebuie între 0.01 și 0.99')
    sys.exit(1)
  
    
# Definim căile către imaginile și label-urile sursă
input_image_path = data_path / "images"
input_label_path = data_path / "labels"

if not input_image_path.exists():
    print("❌ Folder images/ nu există")
    sys.exit(1)


# ---------------------------------------------------------
# DESTINAȚII FIXE ÎN PROJECT_ROOT
# Creăm structura de foldere train/validation
# ---------------------------------------------------------
train_img_path = PROJECT_ROOT / "data/train/images"
train_txt_path = PROJECT_ROOT / "data/train/labels"
val_img_path   = PROJECT_ROOT / "data/validation/images"
val_txt_path   = PROJECT_ROOT / "data/validation/labels"


# Creăm folderele dacă nu există deja
for p in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SPLIT DATASET
# Amestecăm imaginile și le împărțim în train și validation
# ---------------------------------------------------------
img_files = list(input_image_path.glob("*.*"))
random.shuffle(img_files)

train_num = int(len(img_files) * train_percent)
train_imgs = img_files[:train_num]
val_imgs   = img_files[train_num:]


# =============================
# FUNCȚIE COPIERE
# Copiază imaginile și label-urile corespunzătoare
# =============================
def copy_files(img_list, img_dst, lbl_dst):
    for img_path in img_list:
        lbl_path = input_label_path / f"{img_path.stem}.txt"

        shutil.copy2(img_path, img_dst / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_dst / lbl_path.name)


# Copiem fișierele în folderele train și validation
copy_files(train_imgs, train_img_path, train_txt_path)
copy_files(val_imgs, val_img_path, val_txt_path)

# =============================
# LOG FINAL
# Afișăm statistici despre split
# =============================
print("✅ Split realizat cu succes")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"Train: {len(train_imgs)} imagini")
print(f"Val:   {len(val_imgs)} imagini")
