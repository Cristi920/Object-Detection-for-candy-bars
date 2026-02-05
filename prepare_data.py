# ---------------------------------------------------------
# Importăm moduele necesare
# ---------------------------------------------------------
from pathlib import Path
import zipfile
import subprocess
import os


# ---------------------------------------------------------
# Setăm directorul principal al proiectului
# Aici trebuie să existe fișierul ZIP și scriptul de split
# ---------------------------------------------------------
PROJECT_ROOT = Path(
    r"C:\Users\clisu\Desktop\New folder"
)


# ---------------------------------------------------------
# Definim calea către arhiva ZIP cu dataset-ul
# ---------------------------------------------------------
ZIP_PATH = PROJECT_ROOT / "data.zip"


# ---------------------------------------------------------
# Definim folderul unde va fi extras dataset-ul
# ---------------------------------------------------------
EXTRACT_PATH = PROJECT_ROOT / "custom_data"


# ---------------------------------------------------------
# 1. Extragem dataset-ul din ZIP dacă nu a fost deja extras
# Verificăm dacă folderul există ca să evităm extrageri repetate
# ---------------------------------------------------------
if not EXTRACT_PATH.exists():
    print("📦 Unzipping dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)


# ---------------------------------------------------------
# 2. Definim calea către scriptul care face split train/validation
# Scriptul trebuie să existe în folderul proiectului
# ---------------------------------------------------------
split_script = PROJECT_ROOT / "train_val_split.py"


# ---------------------------------------------------------
# 3. Rulăm scriptul extern pentru împărțirea dataset-ului
# 90% date pentru train, 10% pentru validation
# ---------------------------------------------------------
print("📂 Splitting dataset...")
subprocess.run([
    "python",
    str(split_script),
    "--datapath", str(EXTRACT_PATH),
    "--train_pct", "0.9"
])
